#!/usr/bin/env python3
"""
Gas Pump Serial Tester

Utility for testing baseline variance detection on listing photos or any images
containing serial numbers. Useful for calibrating gas pump detection thresholds.

Usage:
    python tools/gas_pump_tester.py review/s-l1603.png
    python tools/gas_pump_tester.py review/*.png --interactive
    python tools/gas_pump_tester.py review/s-l1603.png --crop 100,50,400,100
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np


def calculate_baseline_variance(serial_crop: np.ndarray, debug: bool = False,
                                 threshold_method: str = 'adaptive') -> float:
    """Calculate baseline variance using linear regression approach.

    This is a standalone version of the algorithm from process_production.py,
    with relaxed size constraints for listing photos.

    Args:
        threshold_method: 'adaptive', 'otsu', or 'both' (try both, pick better)
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

    # Try different threshold methods
    if threshold_method == 'adaptive':
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
    elif threshold_method == 'otsu':
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:  # 'both' - try both and use the one that finds more characters
        binary1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
        _, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Count valid contours for each
        def count_chars(b):
            cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return sum(1 for c in cnts
                      if (cv2.boundingRect(c)[3] > crop_h * 0.15 and
                          cv2.boundingRect(c)[3] < crop_h * 0.95))
        binary = binary1 if count_chars(binary1) >= count_chars(binary2) else binary2

    # Find contours of individual characters
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # More permissive filtering for listing photos
    # Characters should be roughly 15-90% of crop height
    char_contours = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        # Filter: reasonable character size (more permissive than scanner version)
        if ch > crop_h * 0.15 and ch < crop_h * 0.95 and cw > 3 and area > 30:
            char_contours.append((x, y, cw, ch))

    if debug:
        print(f"  Found {len(char_contours)} character contours (need >= 5)")

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
    n = len(centers_x)
    sum_x = np.sum(centers_x)
    sum_y = np.sum(centers_y)
    sum_xy = np.sum(centers_x * centers_y)
    sum_x2 = np.sum(centers_x ** 2)

    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-10:
        return (max(centers_y) - min(centers_y)) / avg_height

    m = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - m * sum_x) / n

    predicted_y = m * centers_x + b
    deviations = centers_y - predicted_y
    std_dev = np.std(deviations)
    normalized_variance = std_dev / avg_height

    if debug:
        print(f"  Avg char height: {avg_height:.1f}px")
        print(f"  Line slope: {m:.4f}")
        print(f"  Std dev from line: {std_dev:.2f}px")
        print(f"  Character Y centers: {[f'{y:.1f}' for y in centers_y]}")
        print(f"  Deviations from line: {[f'{d:.1f}' for d in deviations]}")

    return normalized_variance


def auto_find_serial_region(img: np.ndarray) -> tuple:
    """Try to automatically find the serial number region in a bill image.

    Returns (x, y, w, h) or None if not found.
    """
    h, w = img.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Look for green text (serial numbers are green)
    if len(img.shape) == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Green color range
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours in green regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Look for horizontal rectangular regions (serials are wide and short)
        candidates = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / ch if ch > 0 else 0
            area = cw * ch
            # Serial regions are typically wide and short, with aspect ratio > 3
            if aspect > 3 and area > 500 and ch > 10:
                candidates.append((x, y, cw, ch, area))

        if candidates:
            # Sort by area and take largest
            candidates.sort(key=lambda c: c[4], reverse=True)
            x, y, cw, ch, _ = candidates[0]
            # Add padding
            pad_x, pad_y = int(cw * 0.1), int(ch * 0.3)
            return (max(0, x - pad_x), max(0, y - pad_y),
                    min(w - x + pad_x, cw + 2*pad_x), min(h - y + pad_y, ch + 2*pad_y))

    return None


def interactive_crop(img: np.ndarray, window_name: str = "Select Serial Region") -> tuple:
    """Let user interactively select the serial region."""
    roi = cv2.selectROI(window_name, img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)
    return roi if roi[2] > 0 and roi[3] > 0 else None


def process_image(img_path: Path, crop_coords: tuple = None, interactive: bool = False,
                  debug: bool = False, save_crop: bool = False) -> dict:
    """Process a single image and return baseline variance results."""
    img = cv2.imread(str(img_path))
    if img is None:
        return {'error': f"Could not load image: {img_path}"}

    h, w = img.shape[:2]
    result = {
        'file': img_path.name,
        'size': f"{w}x{h}",
        'baseline_variance': 0.0,
        'method': 'none'
    }

    # Determine crop region
    if crop_coords:
        x, y, cw, ch = crop_coords
        result['method'] = 'manual'
    elif interactive:
        coords = interactive_crop(img)
        if coords is None:
            result['error'] = "No region selected"
            return result
        x, y, cw, ch = coords
        result['method'] = 'interactive'
        result['crop_coords'] = f"{x},{y},{cw},{ch}"
    else:
        # Try auto-detection
        coords = auto_find_serial_region(img)
        if coords:
            x, y, cw, ch = coords
            result['method'] = 'auto'
        else:
            # Fallback: use middle portion of image where serial typically is
            x = int(w * 0.1)
            y = int(h * 0.15)
            cw = int(w * 0.5)
            ch = int(h * 0.15)
            result['method'] = 'fallback'

    # Extract and analyze crop
    crop = img[y:y+ch, x:x+cw]

    if debug:
        print(f"\n{img_path.name}:")
        print(f"  Crop region: x={x}, y={y}, w={cw}, h={ch}")
        print(f"  Method: {result['method']}")

    variance = calculate_baseline_variance(crop, debug=debug)
    result['baseline_variance'] = variance

    # Save crop if requested
    if save_crop:
        crop_dir = img_path.parent / "serial_crops"
        crop_dir.mkdir(exist_ok=True)
        crop_path = crop_dir / f"{img_path.stem}_serial.jpg"
        cv2.imwrite(str(crop_path), crop)
        result['crop_saved'] = str(crop_path)

    return result


def main():
    parser = argparse.ArgumentParser(description="Test baseline variance on listing photos")
    parser.add_argument('images', nargs='+', help="Image files to process")
    parser.add_argument('--crop', '-c', help="Manual crop coordinates: x,y,w,h")
    parser.add_argument('--interactive', '-i', action='store_true',
                        help="Interactively select serial region")
    parser.add_argument('--debug', '-d', action='store_true', help="Show debug info")
    parser.add_argument('--save-crops', '-s', action='store_true',
                        help="Save extracted serial crops")
    parser.add_argument('--threshold', '-t', type=float, default=0.20,
                        help="Gas pump threshold (default: 0.20)")

    args = parser.parse_args()

    # Parse crop coordinates if provided
    crop_coords = None
    if args.crop:
        try:
            crop_coords = tuple(int(x) for x in args.crop.split(','))
            if len(crop_coords) != 4:
                raise ValueError("Need 4 values: x,y,w,h")
        except Exception as e:
            print(f"Error parsing crop coordinates: {e}")
            sys.exit(1)

    print("=" * 70)
    print("Gas Pump Serial Tester")
    print(f"Threshold: {args.threshold}")
    print("=" * 70)

    results = []
    for img_path_str in args.images:
        for img_path in Path('.').glob(img_path_str):
            if img_path.is_file() and img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                result = process_image(img_path, crop_coords, args.interactive,
                                       args.debug, args.save_crops)
                results.append(result)

    # Print summary
    print("\nResults:")
    print("-" * 70)
    for r in results:
        if 'error' in r:
            print(f"{r['file']}: ERROR - {r['error']}")
        else:
            flag = "GAS PUMP!" if r['baseline_variance'] >= args.threshold else ""
            print(f"{r['file']:30} BL Var: {r['baseline_variance']:.4f}  {flag}")
            if args.debug and 'crop_coords' in r:
                print(f"  Crop coords for reuse: --crop {r['crop_coords']}")

    print("-" * 70)
    gas_pump_count = sum(1 for r in results if r.get('baseline_variance', 0) >= args.threshold)
    print(f"Total: {len(results)} images, {gas_pump_count} potential gas pumps")


if __name__ == '__main__':
    main()
