#!/usr/bin/env python3
"""
Dollar Bill Serial Number Extractor - Hybrid YOLOv8 + EasyOCR
Optimized for high-volume batch processing (2,000-3,000 bills)

Workflow:
1. Template alignment to normalize bill position
2. Extract serial number regions from aligned bill
3. YOLOv8 detects precise serial bounding boxes in crops
4. EasyOCR reads text from detected regions
5. Fancy number detection
6. CSV output for review
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


class BillAligner:
    """Aligns scanned bills to a reference template."""

    def __init__(self, reference_path):
        self.reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        if self.reference is None:
            raise ValueError(f"Could not load reference image: {reference_path}")

        self.orb = cv2.ORB_create(1000)
        self.ref_kp, self.ref_desc = self.orb.detectAndCompute(self.reference, None)

    def align_image(self, image_path):
        """Align a bill image to the reference."""
        img_color = cv2.imread(str(image_path))
        if img_color is None:
            return None

        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orb.detectAndCompute(img_gray, None)

        if desc is None or len(kp) < 10:
            return img_color  # Return original if alignment fails

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(self.ref_desc, desc, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 10:
            return img_color  # Return original if not enough matches

        ref_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in good_matches])
        img_pts = np.float32([kp[m.trainIdx].pt for m in good_matches])

        M, inliers = cv2.estimateAffinePartial2D(img_pts, ref_pts, method=cv2.RANSAC)

        if M is None:
            return img_color  # Return original if transformation fails

        aligned = cv2.warpAffine(
            img_color, M, (self.reference.shape[1], self.reference.shape[0]),
            flags=cv2.INTER_CUBIC
        )

        return aligned


class FancyNumberDetector:
    """Detects fancy serial numbers (repeaters, radars, etc.)"""

    @staticmethod
    def extract_digits(serial):
        """Extract just the 8 digits from serial (remove letters)"""
        # Serial format: Letter + 8 digits + Letter
        if len(serial) >= 10:
            return serial[1:9]  # Get middle 8 digits
        return None

    @staticmethod
    def is_repeater(digits):
        """Check if first 4 digits repeat (e.g., 12341234)"""
        if len(digits) != 8:
            return False
        return digits[:4] == digits[4:]

    @staticmethod
    def is_radar(digits):
        """Check if it's a palindrome (e.g., 12344321)"""
        if len(digits) != 8:
            return False
        return digits == digits[::-1]

    @staticmethod
    def is_solid(digits):
        """Check if all digits are the same (e.g., 11111111)"""
        if len(digits) != 8:
            return False
        return len(set(digits)) == 1

    @staticmethod
    def is_ladder(digits):
        """Check if digits are sequential (e.g., 12345678 or 87654321)"""
        if len(digits) != 8:
            return False
        nums = [int(d) for d in digits]
        # Ascending
        if all(nums[i] + 1 == nums[i + 1] for i in range(7)):
            return True
        # Descending
        if all(nums[i] - 1 == nums[i + 1] for i in range(7)):
            return True
        return False

    @staticmethod
    def is_low_serial(digits):
        """Check if serial number is very low (< 100)"""
        if len(digits) != 8:
            return False
        try:
            return int(digits) <= 100
        except:
            return False

    @staticmethod
    def is_binary(digits):
        """Check if only contains 0s and 1s"""
        if len(digits) != 8:
            return False
        return set(digits).issubset({'0', '1'})

    @classmethod
    def classify(cls, serial):
        """
        Classify a serial number and return all fancy types found.
        Returns list of fancy types or empty list if not fancy.
        """
        digits = cls.extract_digits(serial)
        if not digits:
            return []

        fancy_types = []

        if cls.is_solid(digits):
            fancy_types.append("SOLID")
        if cls.is_repeater(digits):
            fancy_types.append("REPEATER")
        if cls.is_radar(digits):
            fancy_types.append("RADAR")
        if cls.is_ladder(digits):
            fancy_types.append("LADDER")
        if cls.is_low_serial(digits):
            fancy_types.append("LOW_SERIAL")
        if cls.is_binary(digits):
            fancy_types.append("BINARY")

        return fancy_types


class BillProcessor:
    """Process dollar bills using hybrid template alignment + YOLOv8 + OCR"""

    def __init__(self, yolo_model_path, reference_image_path, use_gpu=False):
        """
        Initialize the processor.

        Args:
            yolo_model_path: Path to your trained YOLOv8 model (best.pt)
            reference_image_path: Path to reference bill for alignment
            use_gpu: Whether to use GPU for processing
        """
        print(f"Loading YOLOv8 model from: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)

        print(f"Loading template aligner...")
        self.aligner = BillAligner(reference_image_path)

        print(f"Loading EasyOCR (GPU={use_gpu})...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)

        self.fancy_detector = FancyNumberDetector()

        print("✓ Ready to process bills!\n")

    def extract_serial_from_crop(self, crop_image):
        """Extract serial number from a cropped region using OCR."""
        # Run OCR on the small cropped region
        results = self.ocr_reader.readtext(
            crop_image,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*',
            detail=1
        )

        pattern = r'[A-Z]\d{8}[A-Z*]'

        for (bbox, text, conf) in results:
            text_clean = re.sub(r'[^A-Z0-9*]', '', text.upper())

            # Check exact match
            match = re.search(pattern, text_clean)
            if match:
                return match.group(0), conf

            # Handle OCR errors (last digit → letter)
            if re.match(r'^[A-Z]\d{9}$', text_clean):
                last_digit = text_clean[-1]
                corrections = {
                    '0': ['O', 'Q'],
                    '1': ['I'],
                    '5': ['S'],
                    '6': ['G'],
                    '8': ['B'],
                }

                if last_digit in corrections:
                    for letter in corrections[last_digit]:
                        corrected = text_clean[:-1] + letter
                        if re.match(pattern, corrected):
                            return corrected, conf

        return None, 0

    def process_single_bill(self, image_path, debug=False):
        """
        Process a single bill image using hybrid approach.

        Returns:
            dict with: filename, serial, fancy_types, confidence, processing_time
        """
        start_time = time.time()

        # Step 1: Align the image to normalize position and scale
        aligned_img = self.aligner.align_image(image_path)

        if aligned_img is None:
            return {
                'filename': Path(image_path).name,
                'serial': None,
                'fancy_types': [],
                'confidence': 0,
                'time': 0,
                'error': 'Failed to load image'
            }

        # Step 2: Run YOLO on the full aligned image
        # (The model was trained on full bills, not just serial crops)
        results = self.yolo_model(aligned_img, verbose=False)

        serials_found = []
        h, w = aligned_img.shape[:2]

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Expand bounding box to capture letters (add 30% padding on left/right, 10% top/bottom)
                # YOLO was trained on tight boxes around digits, need to expand to get letters
                box_width = x2 - x1
                box_height = y2 - y1

                padding_x = int(box_width * 0.30)
                padding_y = int(box_height * 0.10)

                x1_exp = max(0, x1 - padding_x)
                y1_exp = max(0, y1 - padding_y)
                x2_exp = min(w, x2 + padding_x)
                y2_exp = min(h, y2 + padding_y)

                # Crop the expanded serial number region
                crop = aligned_img[y1_exp:y2_exp, x1_exp:x2_exp]

                # Extract text with OCR
                serial, conf = self.extract_serial_from_crop(crop)

                if serial:
                    serials_found.append((serial, conf))

        # Get the best serial (highest confidence)
        if serials_found:
            serial, confidence = max(serials_found, key=lambda x: x[1])
            fancy_types = self.fancy_detector.classify(serial)

            elapsed = time.time() - start_time

            result = {
                'filename': Path(image_path).name,
                'serial': serial,
                'fancy_types': fancy_types,
                'confidence': round(confidence, 2),
                'time': round(elapsed, 2),
                'is_fancy': len(fancy_types) > 0
            }

            if debug:
                fancy_str = ", ".join(fancy_types) if fancy_types else "Not fancy"
                print(f"{result['filename']}: {serial} [{fancy_str}] ({elapsed:.2f}s)")

            return result
        else:
            elapsed = time.time() - start_time
            return {
                'filename': Path(image_path).name,
                'serial': None,
                'fancy_types': [],
                'confidence': 0,
                'time': round(elapsed, 2),
                'error': 'No serial number detected'
            }

    def process_batch(self, image_folder, pattern="Dollar_*.jpg", output_csv="results.csv"):
        """
        Process all bills in a folder.

        Args:
            image_folder: Folder containing scanned bills
            pattern: Glob pattern for images (default: Dollar_*.jpg)
            output_csv: Output CSV file path
        """
        folder = Path(image_folder)
        image_files = sorted(folder.glob(pattern))

        # Process odd numbers only (face bills)
        face_bills = [f for f in image_files if int(f.stem.split('_')[1]) % 2 == 1]

        total = len(face_bills)
        print(f"Processing {total} bills from: {folder}")
        print("=" * 70)

        results = []
        fancy_count = 0
        start_time = time.time()

        for i, image_file in enumerate(face_bills, 1):
            result = self.process_single_bill(image_file, debug=True)
            results.append(result)

            if result.get('is_fancy'):
                fancy_count += 1

            # Progress update every 50 bills
            if i % 50 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = (total - i) * avg_time
                print(f"\nProgress: {i}/{total} bills ({i/total*100:.1f}%) - ETA: {remaining/60:.1f} min")

        # Summary
        total_time = time.time() - start_time
        avg_time = total_time / total if total > 0 else 0
        success_count = sum(1 for r in results if r.get('serial'))

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total bills processed: {total}")
        print(f"Successfully extracted: {success_count} ({success_count/total*100:.1f}%)")
        print(f"Fancy numbers found: {fancy_count}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average time per bill: {avg_time:.2f} seconds")
        print(f"Processing rate: {total/total_time*60:.1f} bills/minute")

        # Save to CSV
        output_path = folder / output_csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'serial', 'fancy_types', 'confidence', 'time', 'is_fancy', 'error'])
            writer.writeheader()
            writer.writerows(results)

        print(f"\nResults saved to: {output_path}")

        # Show fancy numbers
        if fancy_count > 0:
            print("\n" + "=" * 70)
            print("FANCY NUMBERS FOUND:")
            print("=" * 70)
            for r in results:
                if r.get('is_fancy'):
                    fancy_str = ", ".join(r['fancy_types'])
                    print(f"{r['serial']} - {fancy_str}")

        return results


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python process_bills_yolo.py <path_to_yolo_model.pt> [image_folder]")
        print("")
        print("Example:")
        print("  python process_bills_yolo.py best.pt")
        print("  python process_bills_yolo.py best.pt /path/to/scans")
        print("")
        print("After training in Roboflow:")
        print("  1. Export model as 'YOLOv8'")
        print("  2. Download the .pt file (usually 'best.pt')")
        print("  3. Run this script with the path to that file")
        return

    yolo_model_path = sys.argv[1]

    if not Path(yolo_model_path).exists():
        print(f"Error: Model file not found: {yolo_model_path}")
        return

    # Optional: specify image folder
    image_folder = sys.argv[2] if len(sys.argv) > 2 else Path(__file__).parent

    # Use first dollar bill as reference for alignment
    reference_path = Path(image_folder) / "Dollar_01.jpg"
    if not reference_path.exists():
        print(f"Error: Reference image not found: {reference_path}")
        print("Please ensure Dollar_01.jpg exists in the scan folder")
        return

    # Initialize processor (set use_gpu=True if you have NVIDIA GPU)
    processor = BillProcessor(yolo_model_path, reference_path, use_gpu=False)

    # Process all bills
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"serial_numbers_{timestamp}.csv"

    processor.process_batch(image_folder, output_csv=output_csv)


if __name__ == "__main__":
    main()
