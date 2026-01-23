#!/usr/bin/env python3
"""
Dollar Bill Production Pipeline - Phase 1
Automated processing of scanned dollar bills with fancy number detection and cropping.

Features:
- Scanner-agnostic input (auto-detects naming conventions)
- Auto-straightening using printed border detection (no reference image needed)
- Front/back detection using YOLO serial count
- Smart pairing of front/back images
- Percentage-based cropping (scanner-independent)
- Fancy number filtering (only crops fancy bills)
- Stack position tracking
- Batch cleanup helper

Usage:
    python process_production.py /path/to/scans --output fancy_bills/
    python process_production.py /path/to/scans --output fancy_bills/ --all  # crop all bills
"""

import cv2
import numpy as np
import easyocr
import re
from pathlib import Path
from ultralytics import YOLO
import csv
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import argparse
import shutil
import yaml

# Import v2 pattern engine
from pattern_engine_v2 import PatternEngine


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================

class Config:
    """Loads and manages configuration from config.yaml (crop settings only).

    Pattern definitions are now in patterns_v2.yaml, managed by PatternEngine.
    """

    # Default values (used if config.yaml is missing)
    DEFAULT_CROP_REGIONS = {
        'front': {
            'full':   {'x': 0.0,   'y': 0.0,   'w': 1.0,   'h': 1.0},
            'seal':   {'x': 0.586, 'y': 0.183, 'w': 0.254, 'h': 0.537},
            'left':   {'x': 0.0,   'y': 0.0,   'w': 0.384, 'h': 1.0},
            'center': {'x': 0.354, 'y': 0.0,   'w': 0.322, 'h': 1.0},
            'right':  {'x': 0.627, 'y': 0.0,   'w': 0.373, 'h': 1.0},
        },
        'back': {
            'full':   {'x': 0.0,   'y': 0.0,   'w': 1.0,   'h': 1.0},
            'seal':   {'x': 0.619, 'y': 0.221, 'w': 0.261, 'h': 0.557},
            'left':   {'x': 0.0,   'y': 0.0,   'w': 0.384, 'h': 1.0},
            'center': {'x': 0.354, 'y': 0.0,   'w': 0.322, 'h': 1.0},
            'right':  {'x': 0.627, 'y': 0.0,   'w': 0.373, 'h': 1.0},
        }
    }

    DEFAULT_CROP_ORDER = [
        ['front', 'seal'], ['front', 'full'], ['front', 'left'],
        ['front', 'center'], ['front', 'right'], ['back', 'seal'],
        ['back', 'full'], ['back', 'left'], ['back', 'center'], ['back', 'right']
    ]

    def __init__(self, config_path: Optional[Path] = None, patterns_path: Optional[Path] = None):
        self.config_path = config_path
        self.data = {}

        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                self.data = yaml.safe_load(f) or {}
            print(f"Loaded config: {config_path}")
        else:
            print("Using default configuration")

    @property
    def crop_regions(self) -> dict:
        """Get crop regions from config or defaults."""
        if 'crops' in self.data:
            regions = {}
            for side in ['front', 'back']:
                if side in self.data['crops']:
                    regions[side] = {}
                    for name, coords in self.data['crops'][side].items():
                        regions[side][name] = {
                            'x': coords.get('x', 0),
                            'y': coords.get('y', 0),
                            'w': coords.get('w', 1),
                            'h': coords.get('h', 1)
                        }
            return regions
        return self.DEFAULT_CROP_REGIONS

    @property
    def crop_order(self) -> list:
        """Get crop order from config or defaults."""
        if 'crop_order' in self.data:
            return [tuple(item) for item in self.data['crop_order']]
        return [tuple(item) for item in self.DEFAULT_CROP_ORDER]

    @property
    def jpeg_quality(self) -> int:
        """Get JPEG quality setting."""
        if 'options' in self.data:
            return self.data['options'].get('jpeg_quality', 95)
        return 95


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BillPair:
    """Represents a front/back pair of bill scans."""
    front_path: Path
    back_path: Optional[Path]
    stack_position: int  # Physical position in scanned stack
    serial: Optional[str] = None
    fancy_types: list = field(default_factory=list)
    confidence: float = 0.0
    is_fancy: bool = False
    error: Optional[str] = None


# =============================================================================
# CLASSES FROM process_bills_yolo.py (reused)
# =============================================================================

class BillAligner:
    """Aligns scanned bills by detecting the bill's rectangular contour."""

    def __init__(self):
        """No reference image needed - uses contour detection."""
        pass

    def detect_rotation_angle(self, img_gray: np.ndarray) -> float:
        """Detect rotation angle by finding the bill's rectangular contour."""
        h, w = img_gray.shape[:2]

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

        # Use adaptive threshold to handle varying backgrounds
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

        # Also try Otsu's threshold and combine
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        combined = cv2.bitwise_or(thresh, otsu)

        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        # Find the largest contour (should be the bill)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]  # Angle in degrees

        # minAreaRect returns angles in range [-90, 0)
        # We need to normalize based on the rectangle dimensions
        rect_w, rect_h = rect[1]

        # If width < height, the rectangle is more vertical, adjust angle
        if rect_w < rect_h:
            angle = angle + 90

        # Normalize to [-45, 45] range
        if angle > 45:
            angle = angle - 90
        elif angle < -45:
            angle = angle + 90

        return angle

    def align_image(self, image_path) -> Optional[np.ndarray]:
        """Straighten a bill image by detecting its rectangular contour. Returns color image."""
        img_color = cv2.imread(str(image_path))
        if img_color is None:
            return None

        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # Detect rotation angle from contour
        angle = self.detect_rotation_angle(img_gray)

        # Skip rotation if angle is negligible
        if abs(angle) < 0.2:
            return img_color

        # Rotate the image to straighten it
        h, w = img_color.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix - rotate within original bounds
        # This avoids fill artifacts at corners
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Apply rotation with white background for any edge pixels
        aligned = cv2.warpAffine(img_color, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(255, 255, 255))
        return aligned


# =============================================================================
# SCANNER FORMAT DETECTION
# =============================================================================

class ScannerFormatDetector:
    """Detects scanner naming conventions and pairs front/back images."""

    @staticmethod
    def detect_format(image_folder: Path) -> str:
        """
        Detect the scanner naming format.
        Returns: 'suffix' (has _b files) or 'sequential' (numbered pairs)
        """
        files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.jpeg"))

        # Check for _b suffix pattern
        for f in files:
            if '_b.' in f.name.lower() or '_b.' in f.stem.lower():
                return 'suffix'

        return 'sequential'

    @staticmethod
    def find_pairs_suffix(image_folder: Path) -> list[BillPair]:
        """
        Find front/back pairs using suffix naming (e.g., 0001.jpg + 0001_b.jpg)
        """
        files = sorted(image_folder.glob("*.jpg")) + sorted(image_folder.glob("*.jpeg"))
        pairs = []
        seen_bases = set()

        for f in files:
            # Skip back images in first pass
            if '_b.' in f.name.lower():
                continue

            base_name = f.stem
            back_patterns = [
                f.parent / f"{base_name}_b{f.suffix}",
                f.parent / f"{base_name}_B{f.suffix}",
            ]

            back_path = None
            for bp in back_patterns:
                if bp.exists():
                    back_path = bp
                    break

            if base_name not in seen_bases:
                seen_bases.add(base_name)
                pairs.append(BillPair(
                    front_path=f,
                    back_path=back_path,
                    stack_position=len(pairs) + 1
                ))

        return pairs

    @staticmethod
    def find_pairs_sequential(image_folder: Path) -> list[BillPair]:
        """
        Find front/back pairs using sequential numbering.
        Assumes files are sorted and alternating (front, back, front, back...).
        Will verify using YOLO detection later.
        """
        files = sorted(
            list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.jpeg")),
            key=lambda x: x.name
        )

        pairs = []
        i = 0
        while i < len(files):
            front = files[i]
            back = files[i + 1] if i + 1 < len(files) else None

            pairs.append(BillPair(
                front_path=front,
                back_path=back,
                stack_position=len(pairs) + 1
            ))
            i += 2

        return pairs

    @classmethod
    def find_pairs(cls, image_folder: Path) -> tuple[str, list[BillPair]]:
        """
        Detect format and find all front/back pairs.
        Returns: (format_name, list of BillPair)
        """
        fmt = cls.detect_format(image_folder)

        if fmt == 'suffix':
            pairs = cls.find_pairs_suffix(image_folder)
        else:
            pairs = cls.find_pairs_sequential(image_folder)

        return fmt, pairs


# =============================================================================
# PRODUCTION PROCESSOR
# =============================================================================

class ProductionProcessor:
    """Main processor for production pipeline."""

    def __init__(self, yolo_model_path: Path, use_gpu: bool = False, cfg: Optional[Config] = None,
                 patterns_v2_path: Optional[Path] = None):
        self.cfg = cfg or Config()  # Use provided config or create default

        print(f"Loading YOLOv8 model: {yolo_model_path}")
        self.yolo_model = YOLO(str(yolo_model_path))

        print(f"Loading border-based aligner...")
        self.aligner = BillAligner()  # No reference needed - uses border detection

        print(f"Loading EasyOCR (GPU={use_gpu})...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)

        # Use v2 pattern engine (single YAML config)
        print(f"Loading pattern engine v2...")
        self.pattern_engine = PatternEngine(patterns_v2_path)
        pattern_count = len(self.pattern_engine.patterns)
        print(f"  Loaded {pattern_count} patterns from {self.pattern_engine.config_path}")
        print("Ready!\n")

    def count_serial_detections(self, image_path: Path) -> int:
        """Count how many serial numbers YOLO detects in an image."""
        img = cv2.imread(str(image_path))
        if img is None:
            return 0

        # Use lower confidence threshold to handle scans with colored backgrounds
        results = self.yolo_model(img, verbose=False, conf=0.1)
        count = 0
        for result in results:
            count += len(result.boxes)
        return count

    def is_front_image(self, image_path: Path) -> bool:
        """Determine if an image is a front (has serial numbers)."""
        return self.count_serial_detections(image_path) >= 1

    def verify_and_swap_pairs(self, pairs: list[BillPair]) -> list[BillPair]:
        """
        Verify front/back assignments and swap if needed.
        Front images have serial numbers, backs don't.
        """
        verified = []
        for pair in pairs:
            front_is_front = self.is_front_image(pair.front_path)
            back_is_front = pair.back_path and self.is_front_image(pair.back_path)

            # Swap if needed
            if not front_is_front and back_is_front:
                pair.front_path, pair.back_path = pair.back_path, pair.front_path
            elif not front_is_front and not back_is_front:
                # Neither has serials - might be two backs, mark as error
                pair.error = "No serial detected in either image"

            verified.append(pair)
        return verified

    def extract_serial_from_crop(self, crop_image) -> tuple[Optional[str], float]:
        """Extract serial number from a cropped region using OCR."""
        # Try OCR on original + CLAHE enhanced variant for better accuracy
        variants = [crop_image]  # Original

        # CLAHE contrast enhancement helps with some letter recognition
        if len(crop_image.shape) == 3:
            gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop_image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        variants.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))

        pattern = r'[A-L]\d{8}[A-Y*]'

        # Common OCR digit→letter confusions (ordered by likelihood)
        digit_to_letter = {
            '0': ['O', 'D', 'Q'], '1': ['L', 'I'], '5': ['S'],
            '6': ['G', 'C'], '8': ['B'],
        }

        # Collect ALL valid serial readings from all variants
        valid_serials = []

        for variant in variants:
            results = self.ocr_reader.readtext(
                variant,
                allowlist='ABCDEFGHIJKLMNPQRSTUVWXY0123456789*',
                detail=1
            )

            for (bbox, text, conf) in results:
                text_clean = re.sub(r'[^A-Z0-9*]', '', text.upper())

                # Direct match
                match = re.search(pattern, text_clean)
                if match:
                    valid_serials.append((match.group(0), conf))
                    continue

                # Try digit→letter corrections for last position (10 chars)
                if re.match(r'^[A-L]\d{9}$', text_clean):
                    last_digit = text_clean[-1]
                    if last_digit in digit_to_letter:
                        for letter in digit_to_letter[last_digit]:
                            corrected = text_clean[:-1] + letter
                            if re.match(pattern, corrected):
                                valid_serials.append((corrected, conf))
                                break

                # Try digit→letter corrections for first position (10 chars)
                elif re.match(r'^\d{9}[A-Y*]$', text_clean):
                    first_digit = text_clean[0]
                    if first_digit in digit_to_letter:
                        for letter in digit_to_letter[first_digit]:
                            corrected = letter + text_clean[1:]
                            if re.match(pattern, corrected):
                                valid_serials.append((corrected, conf))
                                break

                # Try both positions (10 digits)
                elif re.match(r'^\d{10}$', text_clean):
                    first_digit = text_clean[0]
                    last_digit = text_clean[-1]
                    if first_digit in digit_to_letter and last_digit in digit_to_letter:
                        for first_letter in digit_to_letter[first_digit]:
                            for last_letter in digit_to_letter[last_digit]:
                                corrected = first_letter + text_clean[1:-1] + last_letter
                                if re.match(pattern, corrected):
                                    valid_serials.append((corrected, conf))
                                    break

                # 9 chars starting with valid letter - could be missing last letter
                # Don't auto-append * as it causes false star note detection
                # The consensus voting will pick up the correct reading from other boxes

                # 9 chars ending with letter - likely missing first letter
                elif re.match(r'^\d{8}[A-Y]$', text_clean):
                    first_digit = text_clean[0]
                    if first_digit in digit_to_letter:
                        for letter in digit_to_letter[first_digit]:
                            corrected = letter + text_clean
                            if re.match(pattern, corrected):
                                valid_serials.append((corrected, conf * 0.9))
                                break

        if not valid_serials:
            return None, 0

        # If only one reading, return it
        if len(valid_serials) == 1:
            return valid_serials[0]

        # Multiple readings - use consensus voting on first letter
        from collections import Counter

        # Group by middle digits (most reliable part)
        digit_groups = {}
        for serial, conf in valid_serials:
            middle = serial[1:9]
            if middle not in digit_groups:
                digit_groups[middle] = []
            digit_groups[middle].append((serial, conf))

        # Use the most common middle-digit pattern
        if digit_groups:
            most_common_middle = max(digit_groups.keys(), key=lambda m: len(digit_groups[m]))
            candidates = digit_groups[most_common_middle]

            # Vote on first letter weighted by confidence
            letter_votes = Counter()
            for serial, conf in candidates:
                letter_votes[serial[0]] += conf

            # Pick highest voted letter
            best_letter = letter_votes.most_common(1)[0][0]

            # Return candidate with that letter
            for serial, conf in candidates:
                if serial[0] == best_letter:
                    return serial, conf

            # Construct if needed
            base = candidates[0][0]
            best_conf = max(c for s, c in candidates)
            return best_letter + base[1:], best_conf

        return max(valid_serials, key=lambda x: x[1])

    def extract_serial(self, image_path: Path) -> tuple[Optional[str], float]:
        """Extract serial number from a bill image."""
        aligned_img = self.aligner.align_image(image_path)
        if aligned_img is None:
            return None, 0

        # Use lower confidence threshold to handle scans with colored backgrounds
        results = self.yolo_model(aligned_img, verbose=False, conf=0.1)
        serials_found = []
        h, w = aligned_img.shape[:2]

        # Letter confusions for consensus voting
        first_letter_alts = {
            'C': 'G', 'G': 'C',  # Most common confusion
            'D': 'O', 'O': 'D',
            'B': 'R', 'R': 'B',
            'I': 'L', 'L': 'I',
        }

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Expand bounding box (40% to catch edge letters)
                box_width = x2 - x1
                box_height = y2 - y1
                padding_x = int(box_width * 0.40)
                padding_y = int(box_height * 0.15)

                x1_exp = max(0, x1 - padding_x)
                y1_exp = max(0, y1 - padding_y)
                x2_exp = min(w, x2 + padding_x)
                y2_exp = min(h, y2 + padding_y)

                crop = aligned_img[y1_exp:y2_exp, x1_exp:x2_exp]
                serial, conf = self.extract_serial_from_crop(crop)

                if serial:
                    serials_found.append((serial, conf))

        if not serials_found:
            # Fallback: Check if this might be a star note with * not detected
            # Re-scan crops for 9-char patterns that could be star notes
            star_candidates = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    box_width = x2 - x1
                    box_height = y2 - y1
                    padding_x = int(box_width * 0.40)
                    padding_y = int(box_height * 0.15)

                    x1_exp = max(0, x1 - padding_x)
                    y1_exp = max(0, y1 - padding_y)
                    x2_exp = min(w, x2 + padding_x)
                    y2_exp = min(h, y2 + padding_y)

                    crop = aligned_img[y1_exp:y2_exp, x1_exp:x2_exp]

                    # Try OCR without the full validation
                    ocr_results = self.ocr_reader.readtext(
                        crop,
                        allowlist='ABCDEFGHIJKLMNPQRSTUVWXY0123456789*',
                        detail=1
                    )

                    for (bbox, text, conf) in ocr_results:
                        text_clean = re.sub(r'[^A-Z0-9*]', '', text.upper())
                        # 9 chars starting with valid Fed letter = potential star note
                        if re.match(r'^[A-L]\d{8}$', text_clean):
                            star_candidates.append((text_clean + '*', conf * 0.85))

            if star_candidates:
                # Use most common reading
                from collections import Counter
                serial_counts = Counter(s for s, c in star_candidates)
                best_serial = serial_counts.most_common(1)[0][0]
                best_conf = max(c for s, c in star_candidates if s == best_serial)
                return best_serial, best_conf

            return None, 0

        # If only one reading, return it
        if len(serials_found) == 1:
            return serials_found[0]

        # Multiple readings - use consensus voting
        # Group by the 8 middle digits (which are most reliable)
        from collections import Counter

        # Extract middle digits (positions 1-8) for grouping
        digit_groups = {}
        for serial, conf in serials_found:
            middle = serial[1:9]  # 8 middle digits
            if middle not in digit_groups:
                digit_groups[middle] = []
            digit_groups[middle].append((serial, conf))

        # Find the most common middle-digit pattern
        most_common_middle = max(digit_groups.keys(), key=lambda m: len(digit_groups[m]))
        candidates = digit_groups[most_common_middle]

        # If all candidates agree on first letter, return highest confidence
        first_letters = set(s[0] for s, c in candidates)
        if len(first_letters) == 1:
            return max(candidates, key=lambda x: x[1])

        # Disagreement on first letter - count votes including confusion alternatives
        letter_votes = Counter()
        for serial, conf in candidates:
            first = serial[0]
            letter_votes[first] += conf  # Weight by confidence
            # Also give partial credit to confusion partner
            if first in first_letter_alts:
                alt = first_letter_alts[first]
                # Check if alt is also a valid Fed letter
                if alt in 'ABCDEFGHIJKL':
                    letter_votes[alt] += conf * 0.3  # Lower weight for alternative

        # Pick the letter with highest weighted votes
        best_letter = letter_votes.most_common(1)[0][0]

        # Return the candidate with that letter (or construct it)
        for serial, conf in candidates:
            if serial[0] == best_letter:
                return serial, conf

        # Construct the serial with the voted letter
        base_serial = candidates[0][0]
        best_conf = max(c for s, c in candidates)
        return best_letter + base_serial[1:], best_conf

    def create_crop(self, image: np.ndarray, side: str, region: str) -> np.ndarray:
        """Create a percentage-based crop from an image."""
        h, w = image.shape[:2]
        crop_regions = self.cfg.crop_regions
        coords = crop_regions[side][region]

        x1 = int(coords['x'] * w)
        y1 = int(coords['y'] * h)
        x2 = int((coords['x'] + coords['w']) * w)
        y2 = int((coords['y'] + coords['h']) * h)

        return image[y1:y2, x1:x2]

    def generate_crops(self, pair: BillPair, output_dir: Path) -> list[Path]:
        """Generate crops for a fancy bill pair based on config."""
        crop_paths = []

        # Use aligned (straightened) images for consistent crops
        front_img = self.aligner.align_image(pair.front_path)
        back_img = self.aligner.align_image(pair.back_path) if pair.back_path else None

        crop_order = self.cfg.crop_order
        jpeg_quality = self.cfg.jpeg_quality

        for i, (side, region) in enumerate(crop_order, 1):
            if side == 'front':
                img = front_img
            else:
                img = back_img
                if img is None:
                    continue

            crop = self.create_crop(img, side, region)
            filename = f"{pair.serial}_{i:02d}.jpg"
            crop_path = output_dir / filename

            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            crop_paths.append(crop_path)

        return crop_paths

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        verify_pairs: bool = True,
        crop_all: bool = False
    ) -> dict:
        """
        Process all bills in a directory.

        Args:
            input_dir: Directory containing scanned bills
            output_dir: Directory for fancy bill crops
            verify_pairs: Whether to verify front/back with YOLO (slower but more accurate)
            crop_all: If True, crop ALL bills regardless of fancy status

        Returns:
            Dictionary with processing results
        """
        start_time = time.time()

        # Detect format and find pairs
        print(f"Scanning directory: {input_dir}")
        scanner_format, pairs = ScannerFormatDetector.find_pairs(input_dir)
        print(f"Detected format: {scanner_format}")
        print(f"Found {len(pairs)} bill pairs\n")

        # Verify front/back assignments if requested
        if verify_pairs:
            print("Verifying front/back assignments...")
            pairs = self.verify_and_swap_pairs(pairs)
            print("Verification complete\n")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each pair
        print("Processing bills...")
        print("=" * 70)

        fancy_bills = []
        all_results = []
        non_fancy_files = []

        for pair in pairs:
            if pair.error:
                all_results.append({
                    'position': pair.stack_position,
                    'front_file': pair.front_path.name,
                    'back_file': pair.back_path.name if pair.back_path else '',
                    'serial': '',
                    'fancy_types': '',
                    'error': pair.error
                })
                continue

            # Extract serial
            serial, confidence = self.extract_serial(pair.front_path)
            pair.serial = serial
            pair.confidence = confidence

            if serial:
                # Check for fancy patterns (skip if --all flag)
                if crop_all:
                    pair.fancy_types = ["ALL"]
                    pair.is_fancy = True
                else:
                    fancy_types = self.pattern_engine.classify_simple(serial)
                    pair.fancy_types = fancy_types
                    pair.is_fancy = len(fancy_types) > 0

                fancy_str = ", ".join(pair.fancy_types) if pair.fancy_types else ""
                status = f"[{fancy_str}]" if pair.fancy_types else ""
                print(f"#{pair.stack_position:3d}: {serial} {status}")

                if pair.is_fancy:
                    fancy_bills.append(pair)
                    # Generate crops
                    self.generate_crops(pair, output_dir)
                else:
                    # Track non-fancy files for potential cleanup
                    non_fancy_files.append(str(pair.front_path))
                    if pair.back_path:
                        non_fancy_files.append(str(pair.back_path))
            else:
                pair.error = "No serial detected"
                print(f"#{pair.stack_position:3d}: [ERROR] No serial detected")

            all_results.append({
                'position': pair.stack_position,
                'front_file': pair.front_path.name,
                'back_file': pair.back_path.name if pair.back_path else '',
                'serial': serial or '',
                'fancy_types': ", ".join(pair.fancy_types),
                'confidence': f"{confidence:.2f}",
                'is_fancy': pair.is_fancy,
                'error': pair.error or ''
            })

        # Calculate stats
        total_time = time.time() - start_time
        total_bills = len(pairs)
        successful = sum(1 for p in pairs if p.serial)
        fancy_count = len(fancy_bills)

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total bills processed: {total_bills}")
        print(f"Successfully extracted: {successful} ({successful/total_bills*100:.1f}%)")
        print(f"Fancy numbers found: {fancy_count}")
        print(f"Processing time: {total_time:.1f} seconds")
        print(f"Rate: {total_bills/total_time*60:.1f} bills/minute")

        # Show fancy bill positions
        if fancy_bills:
            positions = [str(p.stack_position) for p in fancy_bills]
            print(f"\nFancy bills at stack positions: {', '.join(positions)}")
            print("\nFancy bills found:")
            for p in fancy_bills:
                print(f"  #{p.stack_position}: {p.serial} [{', '.join(p.fancy_types)}]")

        # Save results
        self._save_results(input_dir, output_dir, all_results, fancy_bills, non_fancy_files)

        return {
            'total': total_bills,
            'successful': successful,
            'fancy_count': fancy_count,
            'fancy_positions': [p.stack_position for p in fancy_bills],
            'time': total_time
        }

    def _save_results(
        self,
        input_dir: Path,
        output_dir: Path,
        all_results: list,
        fancy_bills: list,
        non_fancy_files: list
    ):
        """Save all output files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Full results CSV
        csv_path = input_dir / f"results_{timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'position', 'front_file', 'back_file', 'serial',
                'fancy_types', 'confidence', 'is_fancy', 'error'
            ])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved: {csv_path}")

        # 2. Fancy bill summary
        summary_path = input_dir / f"summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("FANCY BILL SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            if fancy_bills:
                f.write("Stack positions with fancy bills:\n")
                positions = [str(p.stack_position) for p in fancy_bills]
                f.write(f"  {', '.join(positions)}\n\n")

                f.write("Details:\n")
                for p in fancy_bills:
                    f.write(f"  Position #{p.stack_position}: {p.serial}\n")
                    f.write(f"    Types: {', '.join(p.fancy_types)}\n")
                    f.write(f"    Crops: {output_dir / p.serial}_01.jpg ... _10.jpg\n\n")
            else:
                f.write("No fancy bills found.\n")
        print(f"Summary saved: {summary_path}")

        # 3. Non-fancy files list (for optional cleanup)
        cleanup_path = input_dir / f"non_fancy_files_{timestamp}.txt"
        with open(cleanup_path, 'w') as f:
            f.write("# Non-fancy bill scan files\n")
            f.write("# These can be deleted to save space\n")
            f.write("# Review before deleting!\n\n")
            for filepath in non_fancy_files:
                f.write(f"{filepath}\n")
        print(f"Cleanup list saved: {cleanup_path}")
        print(f"  ({len(non_fancy_files)} files can be deleted to save space)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Production pipeline for dollar bill processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_production.py ./scans --output ./fancy_bills
  python process_production.py ./scans --output ./fancy_bills --config ./config.yaml
  python process_production.py ./scans --output ./fancy_bills --gpu
  python process_production.py ./scans --output ./fancy_bills --all  # crop all bills
        """
    )

    parser.add_argument('input_dir', type=Path,
                        help='Directory containing scanned bill images')
    parser.add_argument('--output', '-o', type=Path, default=Path('./fancy_bills'),
                        help='Output directory for fancy bill crops (default: ./fancy_bills)')
    parser.add_argument('--config', '-c', type=Path, default=None,
                        help='Path to config.yaml (default: config.yaml in script directory)')
    parser.add_argument('--model', '-m', type=Path, default=None,
                        help='Path to YOLO model (default: best.pt in script directory)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for processing')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip front/back verification (faster but less accurate)')
    parser.add_argument('--all', action='store_true',
                        help='Crop ALL bills (skip fancy detection, treat everything as fancy)')

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1

    # Find and load config
    script_dir = Path(__file__).parent
    config_path = args.config or script_dir / "config.yaml"
    patterns_v2_path = script_dir / "patterns_v2.yaml"

    cfg = Config(
        config_path if config_path.exists() else None,
        None  # patterns.txt no longer used - v2 YAML is the source of truth
    )

    # Find model
    model_path = args.model or script_dir / "best.pt"
    if not model_path.exists():
        print(f"Error: YOLO model not found: {model_path}")
        return 1

    # Initialize processor (no reference needed - uses border detection for alignment)
    processor = ProductionProcessor(
        model_path,
        use_gpu=args.gpu,
        cfg=cfg,
        patterns_v2_path=patterns_v2_path if patterns_v2_path.exists() else None
    )

    if args.all:
        print("Mode: Processing ALL bills (--all flag)")

    results = processor.process_directory(
        args.input_dir,
        args.output,
        verify_pairs=not args.no_verify,
        crop_all=args.all
    )

    print(f"\nCrops saved to: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
