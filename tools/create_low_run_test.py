#!/usr/bin/env python3
"""
Synthetic Low-Run Test Bill Generator

Generates synthetic test bill images by compositing YOLO-detected regions from
real donor bills, enabling end-to-end testing of the LOW_RUNS pattern through
the full processing pipeline (YOLO -> OCR -> classification).

YOLO detects regions by position on the bill, not by reading text. So pasting
a real "2021" series_year crop from one bill into the series_year location on
another bill will still be detected by YOLO and read correctly by OCR.

Usage:
    # Scan existing results and report what's achievable
    python tools/create_low_run_test.py --results archive/*/results.csv

    # Generate composites into output directory
    python tools/create_low_run_test.py --results archive/*/results.csv --output-dir test_low_runs/

    # Target a specific low_runs.csv row (0-indexed)
    python tools/create_low_run_test.py --results archive/*/results.csv --output-dir test_low_runs/ --target 0
"""

import sys
import argparse
import csv
import glob
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

# YOLO class indices (from process_production.py - v9 model)
YOLO_CLASSES = {
    'ONE_hashed': 0,
    'back_plate': 1,
    'bill_back': 2,
    'bill_front': 3,
    'denomination': 4,
    'front_plate': 5,
    'seal_f': 6,
    'seal_t': 7,
    'serial_number': 8,
    'series_year': 9,
    'star_symbol': 10,
}

# Reverse mapping for YOLO output
YOLO_ID_TO_NAME = {v: k for k, v in YOLO_CLASSES.items()}


def load_inventory(csv_patterns: list[str]) -> list[dict]:
    """Read all results CSVs and extract bill properties."""
    bills = []
    seen_files = set()

    for pattern in csv_patterns:
        for csv_path in glob.glob(pattern):
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    serial = row.get('serial', '')
                    if len(serial) < 2:
                        continue

                    front_file = row.get('front_file', '')
                    if not front_file or front_file in seen_files:
                        continue
                    if not Path(front_file).exists():
                        continue

                    seen_files.add(front_file)

                    front_plate = row.get('front_plate', '')
                    if front_plate.startswith('FW'):
                        facility = 'FW'
                    elif front_plate:
                        facility = 'DC'
                    else:
                        facility = ''

                    bills.append({
                        'serial': serial,
                        'district': serial[0],
                        'block': serial[-1],
                        'series': row.get('series_year', ''),
                        'facility': facility,
                        'front_file': front_file,
                        'back_file': row.get('back_file', ''),
                        'align_angle': float(row.get('front_align_angle', 0) or 0),
                        'align_flipped': row.get('front_align_flipped', 'False') == 'True',
                        'front_plate': front_plate,
                    })

    return bills


def load_low_runs(csv_path: str) -> list[dict]:
    """Load the low_runs.csv target entries."""
    entries = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                'series': row['series'],
                'district': row['district'],
                'block': row['block'],
                'facility': row['facility'],
                'quantity': row.get('quantity', '?'),
            })
    return entries


def apply_cached_alignment(image_path: str, angle: float, flipped: bool):
    """Apply previously computed alignment (from process_production.py:536-570)."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]

    # Apply rotation if needed (0.8 degree threshold with MAD outlier rejection)
    if abs(angle) >= 0.8:
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


def detect_regions(model, img):
    """Run YOLO and return detections as list of dicts."""
    results = model(img, verbose=False, conf=0.3)
    detections = []
    for result in results:
        for box in result.boxes:
            if hasattr(box, 'cls') and box.cls is not None:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = YOLO_ID_TO_NAME.get(cls_id, f'unknown_{cls_id}')
                detections.append({
                    'class': class_name,
                    'x1': x1, 'y1': y1,
                    'x2': x2, 'y2': y2,
                    'conf': conf,
                })
    return detections


def get_best_box(detections, class_name):
    """Get highest-confidence YOLO box for a class."""
    boxes = [d for d in detections if d['class'] == class_name]
    if not boxes:
        return None
    return max(boxes, key=lambda b: b['conf'])


def composite(base_img, donor_img, base_box, donor_box, pad=5):
    """Paste donor region onto base image at the base_box location."""
    h, w = donor_img.shape[:2]
    dx1 = max(0, donor_box['x1'] - pad)
    dy1 = max(0, donor_box['y1'] - pad)
    dx2 = min(w, donor_box['x2'] + pad)
    dy2 = min(h, donor_box['y2'] + pad)
    donor_crop = donor_img[dy1:dy2, dx1:dx2]

    bh, bw = base_img.shape[:2]
    bx1 = max(0, base_box['x1'] - pad)
    by1 = max(0, base_box['y1'] - pad)
    bx2 = min(bw, base_box['x2'] + pad)
    by2 = min(bh, base_box['y2'] + pad)

    target_w = bx2 - bx1
    target_h = by2 - by1
    if target_w <= 0 or target_h <= 0:
        return base_img

    resized = cv2.resize(donor_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    result = base_img.copy()
    result[by1:by2, bx1:bx2] = resized
    return result


def find_recipe(target, bills):
    """Find a composite recipe for a low_runs target entry.

    Returns dict with:
        base: bill dict (matching district + block)
        series_donor: bill dict or None (if base series matches)
        plate_donor: bill dict or None (if base facility matches)
        steps: number of compositing steps needed
    Or None if no viable base bill found.
    """
    # Find base bills matching district + block (serial is untouched)
    bases = [b for b in bills if b['district'] == target['district']
             and b['block'] == target['block']]

    if not bases:
        return None

    # Score each base by how many fields already match (fewer steps = better)
    best = None
    best_steps = 999

    for base in bases:
        steps = 0
        series_donor = None
        plate_donor = None

        # Check series match
        if base['series'] != target['series']:
            # Need a series donor
            donors = [b for b in bills if b['series'] == target['series']]
            if not donors:
                continue  # Can't satisfy this target with this base
            series_donor = donors[0]
            steps += 1

        # Check facility match
        if base['facility'] != target['facility']:
            # Need a plate donor
            donors = [b for b in bills if b['facility'] == target['facility']
                      and b['facility'] != '']
            if not donors:
                continue
            plate_donor = donors[0]
            steps += 1

        if steps < best_steps:
            best_steps = steps
            best = {
                'base': base,
                'series_donor': series_donor,
                'plate_donor': plate_donor,
                'steps': steps,
            }

    return best


def generate_composite(recipe, target, model, output_dir, yolo_cache):
    """Generate a composite test image from a recipe.

    Args:
        recipe: dict from find_recipe()
        target: low_runs target entry
        model: YOLO model
        output_dir: Path for output
        yolo_cache: dict mapping front_file -> (aligned_img, detections)

    Returns:
        (success: bool, message: str)
    """
    base = recipe['base']

    # Load and align base image (with caching)
    base_key = base['front_file']
    if base_key not in yolo_cache:
        base_img = apply_cached_alignment(base['front_file'], base['align_angle'], base['align_flipped'])
        if base_img is None:
            return False, f"Failed to load base image: {base['front_file']}"
        base_dets = detect_regions(model, base_img)
        yolo_cache[base_key] = (base_img, base_dets)
    base_img, base_dets = yolo_cache[base_key]

    result_img = base_img.copy()
    recipe_steps = []

    # Composite series_year if needed
    if recipe['series_donor']:
        donor = recipe['series_donor']
        donor_key = donor['front_file']
        if donor_key not in yolo_cache:
            donor_img = apply_cached_alignment(donor['front_file'], donor['align_angle'], donor['align_flipped'])
            if donor_img is None:
                return False, f"Failed to load series donor: {donor['front_file']}"
            donor_dets = detect_regions(model, donor_img)
            yolo_cache[donor_key] = (donor_img, donor_dets)
        donor_img, donor_dets = yolo_cache[donor_key]

        base_box = get_best_box(base_dets, 'series_year')
        donor_box = get_best_box(donor_dets, 'series_year')

        if base_box and donor_box:
            result_img = composite(result_img, donor_img, base_box, donor_box)
            recipe_steps.append(
                f"Series: swapped '{base['series']}' -> '{donor['series']}' "
                f"from {donor['serial']} ({donor['front_file']})")
        else:
            missing = []
            if not base_box:
                missing.append('base')
            if not donor_box:
                missing.append('donor')
            return False, f"YOLO failed to detect series_year on {', '.join(missing)}"

    # Composite front_plate if needed
    if recipe['plate_donor']:
        donor = recipe['plate_donor']
        donor_key = donor['front_file']
        if donor_key not in yolo_cache:
            donor_img = apply_cached_alignment(donor['front_file'], donor['align_angle'], donor['align_flipped'])
            if donor_img is None:
                return False, f"Failed to load plate donor: {donor['front_file']}"
            donor_dets = detect_regions(model, donor_img)
            yolo_cache[donor_key] = (donor_img, donor_dets)
        donor_img, donor_dets = yolo_cache[donor_key]

        base_box = get_best_box(base_dets, 'front_plate')
        donor_box = get_best_box(donor_dets, 'front_plate')

        if base_box and donor_box:
            result_img = composite(result_img, donor_img, base_box, donor_box)
            recipe_steps.append(
                f"Plate: swapped '{base['front_plate']}' ({base['facility']}) -> "
                f"'{donor['front_plate']}' ({donor['facility']}) "
                f"from {donor['serial']} ({donor['front_file']})")
        else:
            missing = []
            if not base_box:
                missing.append('base')
            if not donor_box:
                missing.append('donor')
            return False, f"YOLO failed to detect front_plate on {', '.join(missing)}"

    # Create output directory
    dir_name = f"{target['series']}_{target['district']}_{target['block']}_{target['facility']}"
    out_dir = output_dir / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save composite front image
    front_path = out_dir / 'front.jpg'
    cv2.imwrite(str(front_path), result_img)

    # Copy back image (unchanged) with _b suffix for auto-pairing
    back_path = out_dir / 'front_b.jpg'
    if base['back_file'] and Path(base['back_file']).exists():
        shutil.copy2(base['back_file'], back_path)
    else:
        # No back image available - create a placeholder
        placeholder = np.ones((result_img.shape[0], result_img.shape[1], 3), dtype=np.uint8) * 240
        cv2.putText(placeholder, "No back image available", (50, placeholder.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
        cv2.imwrite(str(back_path), placeholder)

    # Write recipe file
    recipe_path = out_dir / 'recipe.txt'
    with open(recipe_path, 'w') as f:
        f.write(f"Target: {target['series']},{target['district']},{target['block']},{target['facility']}\n")
        f.write(f"Quantity: {target['quantity']}M printed\n")
        f.write(f"\nBase bill: {base['serial']}\n")
        f.write(f"  Series: {base['series']}\n")
        f.write(f"  District: {base['district']}, Block: {base['block']}\n")
        f.write(f"  Facility: {base['facility']}\n")
        f.write(f"  Front plate: {base['front_plate']}\n")
        f.write(f"  Source: {base['front_file']}\n")
        f.write(f"\nComposite steps: {recipe['steps']}\n")
        if recipe_steps:
            for step in recipe_steps:
                f.write(f"  - {step}\n")
        else:
            f.write("  (none - natural match)\n")

    return True, dir_name


def print_inventory_report(bills, low_runs):
    """Print a report of available inventory and achievable targets."""
    districts = sorted(set(b['district'] for b in bills))
    series_set = sorted(set(b['series'] for b in bills if b['series']))
    facilities = sorted(set(b['facility'] for b in bills if b['facility']))

    print(f"\n{'='*60}")
    print(f"LOW RUN TEST BILL INVENTORY")
    print(f"{'='*60}\n")

    # Count unique CSV files
    csv_files = set()
    for b in bills:
        front = Path(b['front_file'])
        for parent in front.parents:
            csv_candidate = parent / 'results.csv'
            if csv_candidate.exists():
                csv_files.add(str(csv_candidate))
                break

    print(f"Scanned {len(bills):,} bills")
    print(f"\nAvailable districts: {' '.join(districts)}")
    print(f"Available series:   {' '.join(series_set)}")
    print(f"Available facilities: {' '.join(facilities)}")

    # Breakdown by series
    print(f"\nSeries breakdown:")
    for s in series_set:
        count = sum(1 for b in bills if b['series'] == s)
        print(f"  {s}: {count} bills")

    # Facility breakdown
    print(f"\nFacility breakdown:")
    for fac in facilities:
        count = sum(1 for b in bills if b['facility'] == fac)
        print(f"  {fac}: {count} bills")

    # Check each low_runs target
    print(f"\n{'='*60}")
    print(f"COMPOSITE RECIPES ({len(low_runs)} targets)")
    print(f"{'='*60}\n")

    achievable = 0
    natural = 0

    for i, target in enumerate(low_runs):
        label = f"{target['series']},{target['district']},{target['block']},{target['facility']}"
        recipe = find_recipe(target, bills)

        if recipe is None:
            # Why did it fail?
            has_district_block = any(
                b['district'] == target['district'] and b['block'] == target['block']
                for b in bills)
            if not has_district_block:
                print(f"  [{i:2d}] x {label} — No bill with district {target['district']}, block {target['block']}")
            else:
                has_series = any(b['series'] == target['series'] for b in bills)
                has_facility = any(b['facility'] == target['facility'] and b['facility'] != '' for b in bills)
                reasons = []
                if not has_series:
                    reasons.append(f"no series {target['series']}")
                if not has_facility:
                    reasons.append(f"no {target['facility']} facility")
                print(f"  [{i:2d}] x {label} — Base found but missing: {', '.join(reasons)}")
        else:
            achievable += 1
            base = recipe['base']
            if recipe['steps'] == 0:
                natural += 1
                print(f"  [{i:2d}] * {label} — Natural match: {base['serial']} "
                      f"({base['series']}/{base['facility']})")
            else:
                parts = []
                if recipe['series_donor']:
                    parts.append(f"series from {recipe['series_donor']['serial']}")
                if recipe['plate_donor']:
                    parts.append(f"plate from {recipe['plate_donor']['serial']}")
                print(f"  [{i:2d}] + {label} — Base: {base['serial']} "
                      f"({base['series']}/{base['facility']}) + {', '.join(parts)}")

    print(f"\n{achievable} of {len(low_runs)} targets achievable "
          f"({natural} natural matches, {achievable - natural} composites)")

    return achievable


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic test bills for LOW_RUNS pattern testing')
    parser.add_argument('--results', nargs='+', required=True,
                        help='Glob patterns for results CSV files (e.g., archive/*/results.csv)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for generated test images')
    parser.add_argument('--target', type=int, default=None,
                        help='Target a specific low_runs.csv row (0-indexed)')
    parser.add_argument('--low-runs', type=str, default=None,
                        help='Path to low_runs.csv (default: patterns/user/low_runs.csv)')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    # Load low_runs.csv
    low_runs_path = args.low_runs or str(project_root / 'patterns' / 'user' / 'low_runs.csv')
    if not Path(low_runs_path).exists():
        print(f"Error: low_runs.csv not found at {low_runs_path}")
        sys.exit(1)
    low_runs = load_low_runs(low_runs_path)
    print(f"Loaded {len(low_runs)} low-run targets from {low_runs_path}")

    # Load bill inventory from results CSVs
    bills = load_inventory(args.results)
    if not bills:
        print("Error: No bills found in results CSVs. Check your --results paths.")
        sys.exit(1)

    # Filter to target row if specified
    if args.target is not None:
        if args.target < 0 or args.target >= len(low_runs):
            print(f"Error: --target {args.target} out of range (0-{len(low_runs)-1})")
            sys.exit(1)
        low_runs = [low_runs[args.target]]
        print(f"Targeting row {args.target}: {low_runs[0]}")

    # Print inventory report
    achievable = print_inventory_report(bills, low_runs)

    # Generate composites if output directory specified
    if args.output_dir is None:
        print(f"\nUse --output-dir to generate composite test images.")
        return

    if achievable == 0:
        print("\nNo achievable targets - nothing to generate.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLO model
    model_path = project_root / 'best.pt'
    if not model_path.exists():
        print(f"Error: YOLO model not found at {model_path}")
        sys.exit(1)

    print(f"\nLoading YOLO model from {model_path}...")
    from ultralytics import YOLO
    model = YOLO(str(model_path))

    # Cache for aligned images + YOLO detections
    yolo_cache = {}

    print(f"Generating composites into {output_dir}/\n")

    generated = 0
    failed = 0
    report_lines = []

    for i, target in enumerate(low_runs):
        label = f"{target['series']},{target['district']},{target['block']},{target['facility']}"
        recipe = find_recipe(target, bills)

        if recipe is None:
            report_lines.append(f"SKIP {label} — no viable recipe")
            continue

        success, message = generate_composite(recipe, target, model, output_dir, yolo_cache)
        if success:
            generated += 1
            report_lines.append(f"OK   {label} -> {message}/")
            print(f"  OK  {label} -> {message}/")
        else:
            failed += 1
            report_lines.append(f"FAIL {label} — {message}")
            print(f"  FAIL {label} — {message}")

    # Write summary report
    report_path = output_dir / 'report.txt'
    with open(report_path, 'w') as f:
        f.write(f"LOW RUN TEST BILL GENERATION REPORT\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Source: {len(bills)} bills from results CSVs\n")
        f.write(f"Targets: {len(low_runs)} low-run entries\n")
        f.write(f"Generated: {generated}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Skipped: {len(low_runs) - generated - failed}\n\n")
        for line in report_lines:
            f.write(f"{line}\n")

    print(f"\nGenerated {generated} of {len(low_runs)} test images")
    if failed:
        print(f"({failed} failed - see report.txt)")
    print(f"Report: {report_path}")


if __name__ == '__main__':
    main()
