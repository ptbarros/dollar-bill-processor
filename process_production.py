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
import json
from pathlib import Path
from ultralytics import YOLO
import csv
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import argparse
import shutil
import yaml

# Import v2 pattern engine
from pattern_engine_v2 import PatternEngine


# =============================================================================
# TIMING INSTRUMENTATION
# =============================================================================

class TimingStats:
    """Simple timing tracker for performance analysis."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.reset()

    def reset(self):
        """Reset all timing stats for a new bill."""
        self.bill_start = None
        self.times = {}
        self.yolo_calls = 0
        self.ocr_calls = 0

    def start_bill(self):
        """Start timing a new bill."""
        self.reset()
        self.bill_start = time.time()

    def start(self, label: str):
        """Start timing a section."""
        if self.enabled:
            self.times[f"{label}_start"] = time.time()

    def stop(self, label: str):
        """Stop timing a section."""
        if self.enabled and f"{label}_start" in self.times:
            elapsed = time.time() - self.times[f"{label}_start"]
            # Accumulate time for repeated calls
            if label in self.times:
                self.times[label] += elapsed
            else:
                self.times[label] = elapsed

    def add_yolo_call(self):
        """Increment YOLO call counter."""
        self.yolo_calls += 1

    def add_ocr_call(self):
        """Increment OCR call counter."""
        self.ocr_calls += 1

    def get_summary(self, bill_id: str = "") -> str:
        """Get timing summary string."""
        if not self.enabled or self.bill_start is None:
            return ""

        total = time.time() - self.bill_start
        parts = [f"[TIMING] {bill_id}:"]

        # Add individual timings
        for key in ['align', 'detect', 'ocr', 'crops']:
            if key in self.times:
                parts.append(f"{key}={self.times[key]:.2f}s")

        parts.append(f"yolo_calls={self.yolo_calls}")
        parts.append(f"ocr_calls={self.ocr_calls}")
        parts.append(f"total={total:.2f}s")

        return " | ".join(parts)


# Global timing instance (can be enabled/disabled)
_timing = TimingStats(enabled=True)


def get_timing() -> TimingStats:
    """Get the global timing tracker."""
    return _timing


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
    needs_review: bool = False
    review_reason: Optional[str] = None
    is_upside_down: bool = False  # True if bill was scanned upside-down
    baseline_variance: float = 0.0  # Normalized vertical misalignment for gas pump detection
    star_detected: bool = False  # True if star symbol visually detected by YOLO
    # Cached alignment info to avoid redundant YOLO calls in generate_crops()
    front_align_angle: float = 0.0  # Rotation angle from YOLO alignment
    front_align_flipped: bool = False  # Whether front was flipped 180°


@dataclass
class ReviewItem:
    """Item in the review queue."""
    filename: str
    original_read: Optional[str]
    confidence: float
    review_reason: str
    thumbnail_path: Optional[str] = None
    serial_region_path: Optional[str] = None
    corrected_serial: Optional[str] = None
    reviewed: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            'filename': self.filename,
            'original_read': self.original_read,
            'confidence': self.confidence,
            'review_reason': self.review_reason,
            'thumbnail_path': self.thumbnail_path,
            'serial_region_path': self.serial_region_path,
            'corrected_serial': self.corrected_serial,
            'reviewed': self.reviewed,
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ReviewItem':
        return cls(
            filename=d['filename'],
            original_read=d.get('original_read'),
            confidence=d.get('confidence', 0.0),
            review_reason=d['review_reason'],
            thumbnail_path=d.get('thumbnail_path'),
            serial_region_path=d.get('serial_region_path'),
            corrected_serial=d.get('corrected_serial'),
            reviewed=d.get('reviewed', False),
            timestamp=d.get('timestamp', datetime.now().isoformat())
        )


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


class YOLOBillAligner:
    """Aligns scanned bills using YOLO detection for more accurate results."""

    # YOLO class indices
    YOLO_CLASSES = {
        'bill_front': 2,
        'bill_back': 1,
        'seal_t': 6,  # Treasury seal (right side on correct orientation)
        'seal_f': 5,  # Federal Reserve seal (left side on correct orientation)
    }

    def __init__(self, yolo_model):
        """Initialize with a YOLO model."""
        self.yolo_model = yolo_model
        self.contour_aligner = BillAligner()

    def align_image(self, image_path: Path, check_flip: bool = True) -> tuple[Optional[np.ndarray], dict]:
        """
        Align a bill image using YOLO detection.

        Args:
            image_path: Path to the image file
            check_flip: Whether to check and correct upside-down orientation

        Returns:
            tuple: (aligned_image, info_dict)
            info_dict contains: 'angle', 'flipped', 'bill_detected'
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None, {'error': 'Failed to load image'}

        h, w = img.shape[:2]
        info = {'angle': 0.0, 'flipped': False, 'bill_detected': False}

        # Run YOLO detection
        get_timing().add_yolo_call()
        results = self.yolo_model(img, verbose=False, conf=0.3)

        bill_box = None
        seal_t_box = None
        seal_f_box = None

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                if cls_id == self.YOLO_CLASSES['bill_front']:
                    if bill_box is None or conf > bill_box[4]:
                        bill_box = (x1, y1, x2, y2, conf)
                        info['bill_detected'] = True
                elif cls_id == self.YOLO_CLASSES['bill_back']:
                    if bill_box is None or conf > bill_box[4]:
                        bill_box = (x1, y1, x2, y2, conf)
                        info['bill_detected'] = True
                elif cls_id == self.YOLO_CLASSES['seal_t']:
                    if seal_t_box is None or conf > seal_t_box[4]:
                        seal_t_box = (x1, y1, x2, y2, conf)
                elif cls_id == self.YOLO_CLASSES['seal_f']:
                    if seal_f_box is None or conf > seal_f_box[4]:
                        seal_f_box = (x1, y1, x2, y2, conf)

        # Calculate rotation angle using bill region
        angle = 0.0
        if bill_box:
            x1, y1, x2, y2, _ = bill_box
            # Add padding and crop to bill region
            pad = 20
            crop_x1 = max(0, x1 - pad)
            crop_y1 = max(0, y1 - pad)
            crop_x2 = min(w, x2 + pad)
            crop_y2 = min(h, y2 + pad)

            crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # Use contour detection within the bill region
            angle = self.contour_aligner.detect_rotation_angle(gray)
            info['angle'] = angle

        # Apply rotation if needed
        if abs(angle) >= 0.2:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))

        # Check if bill is upside down using seal positions
        if check_flip and seal_t_box and seal_f_box and bill_box:
            bx1, _, bx2, _, _ = bill_box
            bill_cx = (bx1 + bx2) / 2

            seal_t_cx = (seal_t_box[0] + seal_t_box[2]) / 2
            seal_f_cx = (seal_f_box[0] + seal_f_box[2]) / 2

            # Treasury seal should be on right, Federal Reserve on left
            t_on_right = seal_t_cx > bill_cx
            f_on_left = seal_f_cx < bill_cx

            if not t_on_right or not f_on_left:
                # Bill is upside down - rotate 180 degrees
                img = cv2.rotate(img, cv2.ROTATE_180)
                info['flipped'] = True

        return img, info

    def apply_cached_alignment(self, image_path: Path, angle: float, flipped: bool) -> Optional[np.ndarray]:
        """
        Apply previously computed alignment without running YOLO.

        This reuses alignment info from extract_serial() to avoid redundant YOLO calls
        in generate_crops().

        Args:
            image_path: Path to the image file
            angle: Rotation angle from previous alignment
            flipped: Whether to flip 180 degrees

        Returns:
            Aligned image, or None if loading failed
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        h, w = img.shape[:2]

        # Apply rotation if needed
        if abs(angle) >= 0.2:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))

        # Apply 180 flip if needed
        if flipped:
            img = cv2.rotate(img, cv2.ROTATE_180)

        return img


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
        files = sorted(
            list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.jpeg")),
            key=ScannerFormatDetector._natural_sort_key
        )
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
    def _natural_sort_key(path: Path) -> list:
        """Extract numbers from filename for natural sorting.

        'Dollar_01.jpg' -> ['Dollar_', 1, '.jpg']
        'Dollar_100.jpg' -> ['Dollar_', 100, '.jpg']
        This ensures Dollar_2 comes before Dollar_10.
        """
        import re
        parts = re.split(r'(\d+)', path.name)
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    @staticmethod
    def find_pairs_sequential(image_folder: Path) -> list[BillPair]:
        """
        Find front/back pairs using sequential numbering.
        Assumes files are sorted and alternating (front, back, front, back...).
        Will verify using YOLO detection later.
        """
        files = sorted(
            list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.jpeg")),
            key=ScannerFormatDetector._natural_sort_key
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

    # Multi-pass detection configuration
    DETECTION_PASSES = [
        {'conf': 0.1, 'preprocess': None, 'name': 'standard'},
        {'conf': 0.05, 'preprocess': None, 'name': 'low_conf'},
        {'conf': 0.1, 'preprocess': 'contrast', 'name': 'enhanced'},
        {'conf': 0.05, 'preprocess': 'contrast', 'name': 'enhanced_low'},
        {'conf': 0.1, 'preprocess': 'rotate180', 'name': 'rotated'},
    ]

    # Expanded OCR confusion matrix
    CHAR_CONFUSIONS = {
        # Digit to letter
        '0': ['O', 'D', 'Q'],
        '1': ['L', 'I', 'T'],
        '5': ['S'],
        '6': ['G', 'C'],
        '8': ['B'],
        # Letter to digit (reverse mappings)
        'O': ['0'],
        'D': ['0'],
        'Q': ['0'],
        'L': ['1'],
        'I': ['1'],
        'T': ['1', '7'],
        'S': ['5'],
        'G': ['6'],
        'C': ['6'],
        'B': ['8'],
    }

    # Valid Federal Reserve codes (A-L)
    VALID_FED_CODES = set('ABCDEFGHIJKL')

    # YOLO class indices for Dollar Detective model (10 classes)
    # Set to None to use all classes (backward compatible with single-class model)
    YOLO_CLASSES = {
        'back_plate': 0,
        'bill_back': 1,
        'bill_front': 2,
        'denomination': 3,
        'front_plate': 4,
        'seal_f': 5,       # Federal Reserve Bank seal
        'seal_t': 6,       # Treasury seal
        'serial_number': 7,
        'series_year': 8,
        'star_symbol': 9,
    }

    def __init__(self, yolo_model_path: Path, use_gpu: bool = False, cfg: Optional[Config] = None,
                 patterns_v2_path: Optional[Path] = None):
        self.cfg = cfg or Config()  # Use provided config or create default
        self.use_gpu = use_gpu

        print(f"Loading YOLOv8 model: {yolo_model_path}")
        self.yolo_model = YOLO(str(yolo_model_path))

        # Print model class names and find star class dynamically
        self.star_class_id = None
        if hasattr(self.yolo_model, 'names'):
            print(f"  Model classes: {self.yolo_model.names}")
            for cls_id, name in self.yolo_model.names.items():
                if 'star' in name.lower():
                    self.star_class_id = cls_id
                    print(f"  Star class found: '{name}' (id={cls_id})")
                    break
        if self.star_class_id is None:
            print(f"  Warning: No star class found in model")

        print(f"Loading border-based aligner...")
        self.aligner = BillAligner()  # No reference needed - uses border detection

        # YOLO-based aligner for GUI alignment feature
        self.yolo_aligner = YOLOBillAligner(self.yolo_model)

        print(f"Loading EasyOCR (GPU={use_gpu})...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)

        # Use v2 pattern engine (single YAML config)
        print(f"Loading pattern engine v2...")
        self.pattern_engine = PatternEngine(patterns_v2_path)
        pattern_count = len(self.pattern_engine.patterns)
        print(f"  Loaded {pattern_count} patterns from {self.pattern_engine.config_path}")

        # Review queue for items needing manual review
        self.review_queue: List[ReviewItem] = []

        print("Ready!\n")

    def _add_to_review_queue(self, pair: BillPair, reason: str, output_dir: Optional[Path] = None):
        """Add a bill to the review queue and optionally generate thumbnails."""
        thumbnail_path = None
        serial_region_path = None

        if output_dir:
            review_dir = output_dir / "review"
            review_dir.mkdir(exist_ok=True)

            # Generate thumbnail
            img = cv2.imread(str(pair.front_path))
            if img is not None:
                # Create thumbnail (max 400px wide)
                h, w = img.shape[:2]
                scale = min(400 / w, 300 / h)
                thumb = cv2.resize(img, (int(w * scale), int(h * scale)))
                thumb_filename = f"thumb_{pair.front_path.stem}.jpg"
                thumb_path = review_dir / thumb_filename
                cv2.imwrite(str(thumb_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
                thumbnail_path = str(thumb_path)

                # Extract serial region if we have detections
                aligned = self.aligner.align_image(pair.front_path)
                if aligned is not None:
                    boxes = self._detect_serials_single_pass(aligned, 0.05)
                    if boxes:
                        # Get the first detected region
                        x1, y1, x2, y2, _ = boxes[0]
                        h, w = aligned.shape[:2]
                        padding_x = int((x2 - x1) * 0.40)
                        padding_y = int((y2 - y1) * 0.15)
                        x1_exp = max(0, x1 - padding_x)
                        y1_exp = max(0, y1 - padding_y)
                        x2_exp = min(w, x2 + padding_x)
                        y2_exp = min(h, y2 + padding_y)
                        region = aligned[y1_exp:y2_exp, x1_exp:x2_exp]
                        region_filename = f"serial_{pair.front_path.stem}.jpg"
                        region_path = review_dir / region_filename
                        cv2.imwrite(str(region_path), region, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        serial_region_path = str(region_path)

        item = ReviewItem(
            filename=str(pair.front_path),
            original_read=pair.serial,
            confidence=pair.confidence,
            review_reason=reason,
            thumbnail_path=thumbnail_path,
            serial_region_path=serial_region_path
        )
        self.review_queue.append(item)
        pair.needs_review = True
        pair.review_reason = reason

    def save_review_queue(self, output_path: Path):
        """Save review queue to JSON manifest."""
        manifest = {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'total_items': len(self.review_queue),
            'pending_review': sum(1 for item in self.review_queue if not item.reviewed),
            'items': [item.to_dict() for item in self.review_queue]
        }
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    def load_review_queue(self, input_path: Path) -> List[ReviewItem]:
        """Load review queue from JSON manifest."""
        if not input_path.exists():
            return []
        with open(input_path, 'r') as f:
            manifest = json.load(f)
        return [ReviewItem.from_dict(item) for item in manifest.get('items', [])]

    def validate_serial(self, serial: str) -> tuple[bool, Optional[str]]:
        """Validate serial number format and return (valid, reason if invalid)."""
        if not serial:
            return False, "No serial detected"

        # Check length (should be 10 chars: letter + 8 digits + letter/*)
        if len(serial) != 10:
            return False, f"Invalid length: {len(serial)} (expected 10)"

        # Check first character is valid Federal Reserve code (A-L)
        first_char = serial[0]
        if first_char not in self.VALID_FED_CODES:
            return False, f"Invalid Federal Reserve code: {first_char} (must be A-L)"

        # Check middle 8 characters are digits
        middle = serial[1:9]
        if not middle.isdigit():
            return False, f"Invalid digits in middle: {middle}"

        # Check last character
        last_char = serial[-1]
        if last_char != '*' and last_char not in 'ABCDEFGHIJKLMNPQRSTUVWXY':
            return False, f"Invalid suffix: {last_char}"

        return True, None

    def _preprocess_image(self, img: np.ndarray, method: str) -> np.ndarray:
        """Apply preprocessing to improve detection."""
        if method == 'contrast':
            # CLAHE contrast enhancement
            if len(img.shape) == 3:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                lab = cv2.merge([l, a, b])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                return clahe.apply(img)
        elif method == 'rotate180':
            return cv2.rotate(img, cv2.ROTATE_180)
        elif method == 'sharpen':
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            return cv2.filter2D(img, -1, kernel)
        return img

    def align_for_preview(self, image_path: Path) -> tuple[Optional[np.ndarray], dict]:
        """
        Align an image for GUI preview using YOLO-based detection.

        Args:
            image_path: Path to the image to align

        Returns:
            tuple: (aligned_bgr_image, info_dict)
            info_dict contains 'angle', 'flipped', 'bill_detected'
        """
        return self.yolo_aligner.align_image(Path(image_path))

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
        """Extract serial number from a cropped region using OCR with expanded confusion matrix."""
        pattern = r'[A-L]\d{8}[A-Y*]'

        # Try OCR on original image first
        get_timing().add_ocr_call()
        results = self.ocr_reader.readtext(
            crop_image,
            allowlist='ABCDEFGHIJKLMNPQRSTUVWXY0123456789*',
            detail=1
        )

        valid_serials = []
        for (bbox, text, conf) in results:
            text_clean = re.sub(r'[^A-Z0-9*]', '', text.upper())

            # O at suffix position is almost always a misread Q
            if len(text_clean) == 10 and text_clean[-1] == 'O':
                text_clean = text_clean[:-1] + 'Q'

            # Direct match
            match = re.search(pattern, text_clean)
            if match:
                valid_serials.append((match.group(0), conf))
                continue

            # Apply confusion corrections
            corrected = self._apply_confusion_corrections(text_clean, pattern)
            if corrected:
                valid_serials.append((corrected, conf * 0.95))

        # If we got a good result, return it immediately
        if valid_serials:
            best = max(valid_serials, key=lambda x: x[1])
            if best[1] >= 0.5:
                return best

        # Only try enhanced image if original didn't work well
        if len(crop_image.shape) == 3:
            gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop_image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        get_timing().add_ocr_call()
        results = self.ocr_reader.readtext(
            cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
            allowlist='ABCDEFGHIJKLMNPQRSTUVWXY0123456789*',
            detail=1
        )

        for (bbox, text, conf) in results:
            text_clean = re.sub(r'[^A-Z0-9*]', '', text.upper())

            # O at suffix position is almost always a misread Q
            if len(text_clean) == 10 and text_clean[-1] == 'O':
                text_clean = text_clean[:-1] + 'Q'

            match = re.search(pattern, text_clean)
            if match:
                valid_serials.append((match.group(0), conf))
            else:
                corrected = self._apply_confusion_corrections(text_clean, pattern)
                if corrected:
                    valid_serials.append((corrected, conf * 0.95))

        if not valid_serials:
            return None, 0

        if len(valid_serials) == 1:
            return valid_serials[0]

        # Use voting for best serial
        return self._vote_best_serial(valid_serials, [{} for _ in range(10)])

    def _apply_confusion_corrections(self, text: str, pattern: str) -> Optional[str]:
        """Apply OCR confusion corrections to extract a valid serial."""
        # 10 chars - try fixing both ends
        if re.match(r'^\d{10}$', text):
            first, last = text[0], text[-1]
            first_opts = self.CHAR_CONFUSIONS.get(first, [first])
            # For suffix, try Q first (most common 0-lookalike), skip O (never used)
            last_opts = self.CHAR_CONFUSIONS.get(last, [last])
            if last == '0':
                last_opts = ['Q', 'D']  # Q most likely, O never used
            for fl in first_opts:
                for ll in last_opts:
                    corrected = fl + text[1:-1] + ll
                    if re.match(pattern, corrected):
                        return corrected

        # 10 chars with letter at start - try fixing end
        if re.match(r'^[A-L]\d{9}$', text):
            last = text[-1]
            # For suffix position, try Q first (most common 0-lookalike), skip O (never used)
            suffix_opts = self.CHAR_CONFUSIONS.get(last, [])
            if last == '0':
                suffix_opts = ['Q', 'D']  # Q most likely, O never used
            for letter in suffix_opts:
                corrected = text[:-1] + letter
                if re.match(pattern, corrected):
                    return corrected

        # 10 chars with letter at end - try fixing start
        if re.match(r'^\d{9}[A-Y*]$', text):
            first = text[0]
            for letter in self.CHAR_CONFUSIONS.get(first, []):
                corrected = letter + text[1:]
                if re.match(pattern, corrected):
                    return corrected

        # 9 chars ending with letter - try prepending Fed letter
        if re.match(r'^\d{8}[A-Y*]$', text):
            first = text[0]
            for letter in self.CHAR_CONFUSIONS.get(first, []):
                if letter in self.VALID_FED_CODES:
                    corrected = letter + text
                    if re.match(pattern, corrected):
                        return corrected

        # 9 chars starting with Fed letter - potential star note
        if re.match(r'^[A-L]\d{8}$', text):
            # Only add * if confident this is a star note
            # Leave as None to avoid false star notes
            pass

        return None

    def _vote_best_serial(self, serials: list, char_votes: list) -> tuple[str, float]:
        """Use character-by-character voting to determine best serial."""
        from collections import Counter

        # Group by middle digits
        digit_groups = {}
        for serial, conf in serials:
            middle = serial[1:9]
            if middle not in digit_groups:
                digit_groups[middle] = []
            digit_groups[middle].append((serial, conf))

        if not digit_groups:
            return max(serials, key=lambda x: x[1])

        # Get most common middle pattern
        most_common_middle = max(digit_groups.keys(), key=lambda m: len(digit_groups[m]))
        candidates = digit_groups[most_common_middle]

        # Vote on first character
        first_votes = Counter()
        last_votes = Counter()
        for serial, conf in candidates:
            first_votes[serial[0]] += conf
            last_votes[serial[-1]] += conf

        best_first = first_votes.most_common(1)[0][0]
        best_last = last_votes.most_common(1)[0][0]

        # Validate first letter is valid Fed code
        if best_first not in self.VALID_FED_CODES:
            # Try alternatives
            for alt in self.CHAR_CONFUSIONS.get(best_first, []):
                if alt in self.VALID_FED_CODES:
                    best_first = alt
                    break

        # Construct best serial
        best_serial = best_first + most_common_middle + best_last
        best_conf = max(c for s, c in candidates)

        # Verify it matches pattern
        if not re.match(r'^[A-L]\d{8}[A-Y*]$', best_serial):
            # Fall back to highest confidence match
            return max(candidates, key=lambda x: x[1])

        return best_serial, best_conf

    def _detect_serials_single_pass(self, img: np.ndarray, conf: float) -> list:
        """Run YOLO detection with given confidence threshold.

        Filters for serial_number class (7) if using multi-class model,
        or returns all detections for backward compatibility with single-class model.
        """
        get_timing().add_yolo_call()
        results = self.yolo_model(img, verbose=False, conf=conf)
        boxes = []

        # Check if model has class info (multi-class model)
        serial_class_id = self.YOLO_CLASSES.get('serial_number', None)

        for result in results:
            for box in result.boxes:
                # Filter by class if multi-class model
                if serial_class_id is not None and hasattr(box, 'cls') and box.cls is not None:
                    cls_id = int(box.cls[0])
                    if cls_id != serial_class_id:
                        continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2, float(box.conf[0])))
        return boxes

    def _detect_star_symbol(self, img: np.ndarray, conf: float = 0.1) -> bool:
        """Detect if a star symbol is present in the image.

        Uses YOLO star class for definitive star note detection.
        Returns True if star symbol detected, False otherwise.
        """
        if self.star_class_id is None:
            return False  # Model doesn't support star detection

        get_timing().add_yolo_call()
        results = self.yolo_model(img, verbose=False, conf=conf)

        for result in results:
            for box in result.boxes:
                if hasattr(box, 'cls') and box.cls is not None:
                    cls_id = int(box.cls[0])
                    if cls_id == self.star_class_id:
                        box_conf = float(box.conf[0])
                        if box_conf >= 0.2:  # Accept stars with 0.2+ confidence
                            return True
        return False

    def _detect_objects_by_class(self, img: np.ndarray, class_name: str, conf: float = 0.3) -> list:
        """Detect all objects of a specific class.

        Args:
            img: Image to process
            class_name: Name of class to detect (e.g., 'seal_t', 'front_plate')
            conf: Confidence threshold

        Returns:
            List of (x1, y1, x2, y2, confidence) tuples
        """
        results = self.yolo_model(img, verbose=False, conf=conf)
        boxes = []
        class_id = self.YOLO_CLASSES.get(class_name, None)

        if class_id is None:
            return boxes  # Unknown class

        for result in results:
            for box in result.boxes:
                if hasattr(box, 'cls') and box.cls is not None:
                    cls_id = int(box.cls[0])
                    if cls_id == class_id:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        boxes.append((x1, y1, x2, y2, float(box.conf[0])))
        return boxes

    def _calculate_baseline_variance(self, serial_crop: np.ndarray) -> float:
        """Calculate baseline variance of characters in a serial number crop.

        This detects 'gas pump' printing errors where digits are vertically
        misaligned like an old gas pump display.

        The algorithm fits a line to the character centers to account for image tilt,
        then measures the standard deviation of deviations from that line. This
        distinguishes between tilted-but-aligned serials and true gas pump effect.

        Returns:
            float: Normalized baseline variance (0.0 = perfect alignment, higher = more misaligned)
                   Values > 0.10 typically indicate gas pump effect
        """
        if serial_crop is None or serial_crop.size == 0:
            return 0.0

        crop_h, crop_w = serial_crop.shape[:2]
        if crop_h < 10 or crop_w < 20:
            return 0.0

        # Convert to grayscale
        if len(serial_crop.shape) == 3:
            gray = cv2.cvtColor(serial_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = serial_crop

        # Adaptive threshold to find text pixels
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours of individual characters
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by size (should be character-sized)
        char_contours = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            # Filter: reasonable character size
            if ch > crop_h * 0.3 and ch < crop_h * 0.95 and cw > 5 and area > 50:
                char_contours.append((x, y, cw, ch))

        if len(char_contours) < 5:
            return 0.0

        # Sort by x position (left to right)
        char_contours.sort(key=lambda c: c[0])

        # Calculate center X and Y positions for each character
        centers_x = np.array([x + cw / 2 for x, y, cw, ch in char_contours])
        centers_y = np.array([(y + ch / 2) for x, y, cw, ch in char_contours])
        heights = [ch for x, y, cw, ch in char_contours]
        avg_height = np.mean(heights)

        if avg_height <= 0:
            return 0.0

        # Fit a line to the character centers to account for image tilt
        # y = mx + b (linear regression)
        n = len(centers_x)
        sum_x = np.sum(centers_x)
        sum_y = np.sum(centers_y)
        sum_xy = np.sum(centers_x * centers_y)
        sum_x2 = np.sum(centers_x ** 2)

        # Calculate slope (m) and intercept (b)
        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10:
            # Vertical line or single point - use simple range
            return (max(centers_y) - min(centers_y)) / avg_height

        m = (n * sum_xy - sum_x * sum_y) / denom
        b = (sum_y - m * sum_x) / n

        # Calculate predicted Y for each character along the fitted line
        predicted_y = m * centers_x + b

        # Calculate deviations from the fitted line
        deviations = centers_y - predicted_y

        # Use standard deviation of deviations (not range) for robustness
        # This measures how much characters deviate from the expected baseline
        std_dev = np.std(deviations)

        # Normalize by average character height
        normalized_variance = std_dev / avg_height

        return normalized_variance

    def _extract_serial_from_boxes(self, img: np.ndarray, boxes: list) -> list:
        """Extract serial numbers from detected bounding boxes.

        Returns:
            list of tuples: (serial, ocr_conf, det_conf, baseline_variance)
            baseline_variance is normalized vertical misalignment, for gas pump detection
        """
        serials_found = []
        h, w = img.shape[:2]

        for x1, y1, x2, y2, det_conf in boxes:
            # Early exit: if we already have a high-confidence serial, skip remaining boxes
            # This avoids redundant OCR calls on multiple detected regions
            if serials_found and max(conf for _, conf, _, _ in serials_found) >= 0.7:
                break

            # Expand bounding box (40% to catch edge letters)
            box_width = x2 - x1
            box_height = y2 - y1
            padding_x = int(box_width * 0.40)
            padding_y = int(box_height * 0.15)

            x1_exp = max(0, x1 - padding_x)
            y1_exp = max(0, y1 - padding_y)
            x2_exp = min(w, x2 + padding_x)
            y2_exp = min(h, y2 + padding_y)

            crop = img[y1_exp:y2_exp, x1_exp:x2_exp]
            serial, conf = self.extract_serial_from_crop(crop)

            if serial:
                # Calculate baseline variance for gas pump detection
                # Use the tight crop (not expanded) for better character detection
                tight_crop = img[y1:y2, x1:x2]
                baseline_variance = self._calculate_baseline_variance(tight_crop)
                serials_found.append((serial, conf, det_conf, baseline_variance))

        return serials_found

    def extract_serial(self, image_path: Path) -> tuple[Optional[str], float, bool, float, bool, dict]:
        """
        Extract serial number using multi-pass detection pipeline.

        Returns:
            tuple: (serial, confidence, is_upside_down, baseline_variance, star_detected, align_info)
            - baseline_variance: normalized vertical misalignment of digits (gas pump detection)
            - star_detected: True if star symbol visually detected (definitive star note)
            - align_info: dict with 'angle' and 'flipped' for reuse in generate_crops()
        """
        timing = get_timing()

        # Use YOLO-based alignment for more accurate straightening
        # This improves baseline variance accuracy for gas pump detection
        timing.start('align')
        aligned_img, align_info = self.yolo_aligner.align_image(image_path)
        if aligned_img is None:
            # Fallback to contour-based alignment if YOLO fails
            aligned_img = self.aligner.align_image(image_path)
            if aligned_img is None:
                timing.stop('align')
                return None, 0, False, 0.0, False, {'angle': 0.0, 'flipped': False}
            align_info = {'angle': 0.0, 'flipped': False}
        timing.stop('align')

        # Track if YOLO detected the bill was flipped
        yolo_detected_flip = align_info.get('flipped', False)

        # Track best baseline_variance across all passes
        best_baseline_variance = 0.0

        # Check for star symbol (definitive star note detection)
        timing.start('detect')
        star_detected = self._detect_star_symbol(aligned_img)

        # First pass: standard detection (fastest path for most bills)
        boxes = self._detect_serials_single_pass(aligned_img, conf=0.1)
        if boxes:
            serials = self._extract_serial_from_boxes(aligned_img, boxes)
            if serials:
                # Got results on first pass - use them
                # serials now includes baseline_variance as 4th element
                result = self._consensus_vote([(s, oc, dc, 'standard') for s, oc, dc, hr in serials])
                # Track height ratio from best detection
                best_baseline_variance = max(hr for s, oc, dc, hr in serials)

                # Check for star in OCR result (fallback if YOLO missed it)
                if not star_detected and result[0] and result[0].endswith('*'):
                    star_detected = True

                # Early exit conditions:
                # 1. Good confidence (>= 0.5) - normal case
                # 2. Star detected (YOLO or OCR) + any serial found - star notes are rare,
                #    if we found a star and got a serial, trust it and skip fallbacks
                if result[1] >= 0.5 or (star_detected and result[0]):
                    timing.stop('detect')
                    return result[0], result[1], yolo_detected_flip, best_baseline_variance, star_detected, align_info

        # Only run additional passes if first pass failed or had low confidence
        all_serials = []
        all_baseline_variances = []
        found_via_rotation = False

        for pass_config in self.DETECTION_PASSES[1:]:  # Skip first pass, already done
            conf = pass_config['conf']
            preprocess = pass_config['preprocess']
            pass_name = pass_config['name']

            # Prepare image
            img = aligned_img.copy()
            if preprocess:
                img = self._preprocess_image(img, preprocess)

            # Run detection
            boxes = self._detect_serials_single_pass(img, conf)

            if boxes:
                serials = self._extract_serial_from_boxes(img, boxes)
                if serials:
                    for s, ocr_conf, det_conf, baseline_variance in serials:
                        all_serials.append((s, ocr_conf, det_conf, pass_name))
                        all_baseline_variances.append(baseline_variance)
                        # Check for star in OCR result (fallback if YOLO missed it)
                        if not star_detected and s and s.endswith('*'):
                            star_detected = True

                    # Track if best result came from rotated pass
                    if preprocess == 'rotate180':
                        found_via_rotation = True

                    # If we got good results, stop early
                    # Also exit early for star notes - they often have lower confidence
                    # but if we detected a star and found a serial, that's good enough
                    best_conf = max(oc for s, oc, dc, hr in serials)
                    if best_conf >= 0.6 or (star_detected and best_conf >= 0.3):
                        break

        if not all_serials:
            # Last resort: try whole-image OCR scan for serial patterns
            serial, conf = self._fallback_ocr_scan(aligned_img)
            # Check for star in fallback result
            if not star_detected and serial and serial.endswith('*'):
                star_detected = True
            timing.stop('detect')
            return serial, conf, False, 0.0, star_detected, align_info

        result = self._consensus_vote(all_serials)

        # Use the max height ratio from successful detections
        if all_baseline_variances:
            best_baseline_variance = max(all_baseline_variances)

        # Determine if the bill was upside down
        # YOLO alignment already corrected flip if detected, but track it for reporting
        # Also check if serial was found via rotated pass as fallback
        if all_serials:
            rotated_count = sum(1 for s, oc, dc, name in all_serials if 'rotated' in name)
            normal_count = len(all_serials) - rotated_count
            found_via_rotation = rotated_count > normal_count

        # Bill was upside down if YOLO detected flip OR serial found via rotation
        is_upside_down = yolo_detected_flip or found_via_rotation

        timing.stop('detect')
        return result[0], result[1], is_upside_down, best_baseline_variance, star_detected, align_info

    def _fallback_ocr_scan(self, img: np.ndarray) -> tuple[Optional[str], float]:
        """Fallback: scan entire image for serial patterns."""
        pattern = r'[A-L]\d{8}[A-Y*]'

        # Try OCR on full image
        get_timing().add_ocr_call()
        ocr_results = self.ocr_reader.readtext(
            img,
            allowlist='ABCDEFGHIJKLMNPQRSTUVWXY0123456789*',
            detail=1
        )

        candidates = []
        for (bbox, text, conf) in ocr_results:
            text_clean = re.sub(r'[^A-Z0-9*]', '', text.upper())

            # O at suffix position is almost always a misread Q
            if len(text_clean) == 10 and text_clean[-1] == 'O':
                text_clean = text_clean[:-1] + 'Q'

            match = re.search(pattern, text_clean)
            if match:
                candidates.append((match.group(0), conf))
            # Check for 9-digit star note pattern
            elif re.match(r'^[A-L]\d{8}$', text_clean):
                candidates.append((text_clean + '*', conf * 0.85))

        if candidates:
            # Return highest confidence
            best = max(candidates, key=lambda x: x[1])
            return best

        return None, 0

    def _consensus_vote(self, all_serials: list) -> tuple[Optional[str], float]:
        """Use weighted voting to pick best serial from multiple reads."""
        if len(all_serials) == 1:
            return all_serials[0][0], all_serials[0][1]

        from collections import Counter

        # Letter confusions for consensus voting
        first_letter_alts = {
            'C': 'G', 'G': 'C',
            'D': 'O', 'O': 'D',
            'B': 'R', 'R': 'B',
            'I': 'L', 'L': 'I',
        }

        # Group by the 8 middle digits (most reliable)
        digit_groups = {}
        for serial, ocr_conf, det_conf, pass_name in all_serials:
            middle = serial[1:9]
            if middle not in digit_groups:
                digit_groups[middle] = []
            # Weight by both OCR and detection confidence
            combined_weight = ocr_conf * (det_conf ** 0.5)
            digit_groups[middle].append((serial, combined_weight, pass_name))

        # Find the most common middle-digit pattern
        most_common_middle = max(digit_groups.keys(), key=lambda m: len(digit_groups[m]))
        candidates = digit_groups[most_common_middle]

        # If all agree on first letter, return highest weight
        first_letters = set(s[0] for s, w, p in candidates)
        if len(first_letters) == 1:
            best = max(candidates, key=lambda x: x[1])
            return best[0], best[1]

        # Disagreement - weighted voting on first letter
        letter_votes = Counter()
        for serial, weight, pass_name in candidates:
            first = serial[0]
            letter_votes[first] += weight
            # Partial credit for confusion partners
            if first in first_letter_alts:
                alt = first_letter_alts[first]
                if alt in self.VALID_FED_CODES:
                    letter_votes[alt] += weight * 0.3

        best_letter = letter_votes.most_common(1)[0][0]

        # Return candidate with that letter
        for serial, weight, pass_name in candidates:
            if serial[0] == best_letter:
                return serial, weight

        # Construct with voted letter
        base = candidates[0][0]
        best_weight = max(w for s, w, p in candidates)
        return best_letter + base[1:], best_weight

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
        timing = get_timing()
        timing.start('crops')
        crop_paths = []

        # Use cached alignment for front if available (saves 1 YOLO call)
        # This reuses alignment computed during extract_serial()
        # We know alignment was cached if serial extraction succeeded (pair.serial is set)
        if pair.serial is not None:
            # We have cached alignment info - apply it without YOLO
            front_img = self.yolo_aligner.apply_cached_alignment(
                pair.front_path, pair.front_align_angle, pair.front_align_flipped
            )
            front_flipped = pair.front_align_flipped
        else:
            # No cached info (serial extraction failed) - use YOLO
            front_img, front_info = self.yolo_aligner.align_image(pair.front_path)
            front_flipped = front_info.get('flipped', False)

        # For back, use YOLO alignment but don't check flip (no seals on back)
        # Instead, flip the back if the front was flipped (same physical orientation)
        back_img = None
        if pair.back_path:
            back_img, _ = self.yolo_aligner.align_image(pair.back_path, check_flip=False)
            # If front was flipped, back should be too (same physical bill)
            if front_flipped and back_img is not None:
                back_img = cv2.rotate(back_img, cv2.ROTATE_180)

        # Also handle the is_upside_down flag from serial detection (legacy support)
        if pair.is_upside_down and not front_flipped:
            if front_img is not None:
                front_img = cv2.rotate(front_img, cv2.ROTATE_180)
            if back_img is not None:
                back_img = cv2.rotate(back_img, cv2.ROTATE_180)

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

        timing.stop('crops')
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
            timing = get_timing()
            timing.start_bill()

            if pair.error:
                all_results.append({
                    'position': pair.stack_position,
                    'front_file': pair.front_path.name,
                    'back_file': pair.back_path.name if pair.back_path else '',
                    'serial': '',
                    'fancy_types': '',
                    'confidence': '0.00',
                    'baseline_variance': '0.0000',
                    'star_detected': False,
                    'is_fancy': False,
                    'needs_review': True,
                    'error': pair.error
                })
                self._add_to_review_queue(pair, pair.error, output_dir)
                print(timing.get_summary(f"#{pair.stack_position} ERROR"))
                continue

            # Extract serial using multi-pass detection
            serial, confidence, is_upside_down, baseline_variance, star_detected, align_info = self.extract_serial(pair.front_path)
            pair.confidence = confidence
            pair.is_upside_down = is_upside_down
            pair.baseline_variance = baseline_variance
            pair.star_detected = star_detected
            # Cache alignment info for reuse in generate_crops()
            pair.front_align_angle = align_info.get('angle', 0.0)
            pair.front_align_flipped = align_info.get('flipped', False)

            # If star symbol visually detected but OCR missed it, append '*' to serial
            if serial and star_detected and not serial.endswith('*'):
                serial = serial[:-1] + '*' if len(serial) == 10 else serial + '*'

            pair.serial = serial

            # Validate serial format
            is_valid, validation_error = self.validate_serial(serial)

            if serial and is_valid:
                # Flag low confidence reads for review
                if confidence < 0.5:
                    self._add_to_review_queue(pair, f"Low confidence: {confidence:.2f}", output_dir)

                # Check for fancy patterns (skip if --all flag)
                if crop_all:
                    pair.fancy_types = ["ALL"]
                    pair.is_fancy = True
                else:
                    # Pass baseline_variance for gas pump detection
                    metadata = {'baseline_variance': pair.baseline_variance}
                    fancy_types = self.pattern_engine.classify_simple(serial, metadata)
                    pair.fancy_types = fancy_types
                    pair.is_fancy = len(fancy_types) > 0

                fancy_str = ", ".join(pair.fancy_types) if pair.fancy_types else ""
                review_flag = " [REVIEW]" if pair.needs_review else ""
                status = f"[{fancy_str}]{review_flag}" if pair.fancy_types else review_flag
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
            elif serial and not is_valid:
                # Invalid serial format - needs review
                pair.error = validation_error
                self._add_to_review_queue(pair, f"Validation failed: {validation_error}", output_dir)
                print(f"#{pair.stack_position:3d}: {serial} [INVALID: {validation_error}]")
            else:
                pair.error = "No serial detected"
                self._add_to_review_queue(pair, "No serial detected", output_dir)
                print(f"#{pair.stack_position:3d}: [ERROR] No serial detected")

            all_results.append({
                'position': pair.stack_position,
                'front_file': pair.front_path.name,
                'back_file': pair.back_path.name if pair.back_path else '',
                'serial': serial or '',
                'fancy_types': ", ".join(pair.fancy_types),
                'confidence': f"{confidence:.2f}",
                'baseline_variance': f"{pair.baseline_variance:.4f}",
                'star_detected': pair.star_detected,
                'is_fancy': pair.is_fancy,
                'needs_review': pair.needs_review,
                'error': pair.error or ''
            })

            # Print timing summary for this bill
            bill_id = f"#{pair.stack_position} {serial or 'NO_SERIAL'}"
            print(timing.get_summary(bill_id))

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
                'fancy_types', 'confidence', 'baseline_variance', 'star_detected',
                'is_fancy', 'needs_review', 'error'
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

        # 4. Review queue JSON manifest
        if self.review_queue:
            review_path = input_dir / f"review_queue_{timestamp}.json"
            self.save_review_queue(review_path)
            print(f"Review queue saved: {review_path}")
            print(f"  ({len(self.review_queue)} items need manual review)")


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
