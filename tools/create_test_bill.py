#!/usr/bin/env python3
"""
Synthetic Test Bill Generator — Serial Number Compositing

Generates synthetic bill images with specific serial numbers by compositing
individual character glyphs from real scanned bills. Enables pattern regression
testing through the full pipeline (YOLO → OCR → classification).

Usage:
    # Generate a bill with a specific serial
    python tools/create_test_bill.py --serial "A12344321B" --results archive/*/results.csv --output-dir test_bills/

    # Generate random serials matching a pattern
    python tools/create_test_bill.py --pattern RADAR --count 5 --results archive/*/results.csv --output-dir test_bills/

    # Generate one test bill for every pattern
    python tools/create_test_bill.py --all-patterns --results archive/*/results.csv --output-dir test_bills/

    # Report what's possible without generating (no YOLO needed)
    python tools/create_test_bill.py --pattern RADAR --count 3 --results archive/*/results.csv --dry-run
"""

import sys
import argparse
import random
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from tools.create_low_run_test import (
    load_inventory,
    apply_cached_alignment,
    detect_regions,
    get_best_box,
)

# Patterns that can't be tested via digit compositing alone
SKIP_PATTERNS = {
    'GAS_PUMP',    # Requires physical misalignment measurement
    'STAR',        # Requires star symbol image (★ note)
    'LOW_RUNS',    # Requires metadata (series/district/block)
    'KNOWN_SERIALS',  # Requires external data file match
}


def segment_characters(serial_crop):
    """Segment a serial region into individual character bounds.

    Uses vertical projection to find character columns, then measures
    ink bounds per character via horizontal projection.

    Args:
        serial_crop: BGR or grayscale image of a serial number region.

    Returns:
        List of dicts with keys: x1, x2, width, height, top, bottom
        Sorted left-to-right. Empty list if segmentation fails.
    """
    if serial_crop is None or serial_crop.size == 0:
        return []

    crop_h, crop_w = serial_crop.shape[:2]
    if crop_h < 10 or crop_w < 20:
        return []

    # Convert to grayscale
    if len(serial_crop.shape) == 3:
        gray = cv2.cvtColor(serial_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = serial_crop

    # Binary threshold using Otsu's method (inverted so digits are white)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find character columns using vertical projection
    v_proj = np.sum(binary, axis=0)
    proj_thresh = np.max(v_proj) * 0.1 if np.max(v_proj) > 0 else 0

    in_char = False
    char_bounds = []
    start = 0

    for x in range(crop_w):
        if v_proj[x] > proj_thresh and not in_char:
            start = x
            in_char = True
        elif v_proj[x] <= proj_thresh and in_char:
            char_bounds.append((start, x))
            in_char = False
    if in_char:
        char_bounds.append((start, crop_w - 1))

    # Merge nearby character bounds (handles split "4" and serif fragments)
    merged_bounds = []
    for bound in char_bounds:
        if merged_bounds and bound[0] - merged_bounds[-1][1] < 4:
            merged_bounds[-1] = (merged_bounds[-1][0], bound[1])
        else:
            merged_bounds.append(bound)
    char_bounds = merged_bounds

    # Get vertical extent of each character
    chars = []
    for cx1, cx2 in char_bounds:
        col_strip = binary[:, cx1:cx2]
        h_proj = np.sum(col_strip, axis=1)
        ink_rows = np.where(h_proj > 0)[0]
        if len(ink_rows) > 0:
            chars.append({
                'x1': cx1, 'x2': cx2,
                'width': cx2 - cx1,
                'height': int(ink_rows[-1]) - int(ink_rows[0]),
                'top': int(ink_rows[0]),
                'bottom': int(ink_rows[-1]),
            })

    # Filter out fragments: too narrow (< 5px) or too short (< 50% median height)
    if chars:
        chars = [c for c in chars if c['width'] >= 5]
    if chars:
        heights = [c['height'] for c in chars]
        median_height = np.median(heights)
        chars = [c for c in chars if c['height'] >= median_height * 0.5]

    return chars


def map_characters_to_serial(chars, serial):
    """Map segmented character bounds to known serial characters.

    Args:
        chars: List of character bound dicts from segment_characters().
        serial: Full serial string (e.g., "A12345678B").

    Returns:
        Dict mapping each character to its bound dict, or None if mapping fails.
        E.g., {'A': [bound], '1': [bound], ...} but keyed by (position, char).
        Actually returns list of (char, bound) tuples for the digit portion.
    """
    n = len(chars)
    digits = serial[1:-1]  # 8 digits

    if n == 10:
        # Full serial: position i maps to serial[i]
        return [(serial[i], chars[i]) for i in range(10)]
    elif n == 8:
        # Just digits (prefix/suffix outside YOLO box)
        return [(digits[i], chars[i]) for i in range(8)]
    elif n == 9:
        # One end missing — heuristic: check if first char is narrow (letter)
        # Letters tend to be narrower than digits in serial fonts
        if chars[0]['width'] < chars[1]['width'] * 0.7:
            # First char looks like a letter → prefix present, suffix missing
            return [(serial[i], chars[i]) for i in range(9)]
        else:
            # Suffix present, prefix missing
            return [(serial[i + 1], chars[i]) for i in range(9)]
    else:
        return None


def build_digit_atlas(bills, model, max_bills=50, min_samples=3):
    """Scan bills to collect character crops for each digit/letter.

    Args:
        bills: List of bill dicts from load_inventory().
        model: YOLO model.
        max_bills: Maximum number of bills to scan.
        min_samples: Stop when each digit 0-9 has this many samples.

    Returns:
        Dict mapping character -> list of crop images.
        E.g., {'0': [crop1, crop2], 'A': [crop1], ...}
    """
    atlas = {}
    scanned = 0

    # Shuffle to get variety
    sample_bills = list(bills)
    random.shuffle(sample_bills)

    for bill in sample_bills[:max_bills]:
        # Check if we have enough samples for all digits
        digits_covered = sum(1 for d in '0123456789'
                            if len(atlas.get(d, [])) >= min_samples)
        if digits_covered == 10:
            print(f"  Atlas complete after scanning {scanned} bills")
            break

        # Load and align image
        img = apply_cached_alignment(bill['front_file'],
                                     bill['align_angle'],
                                     bill['align_flipped'])
        if img is None:
            continue

        # Detect serial regions
        detections = detect_regions(model, img)
        serial_boxes = [d for d in detections if d['class'] == 'serial_number']
        if not serial_boxes:
            continue

        # Use highest-confidence serial box
        best_box = max(serial_boxes, key=lambda b: b['conf'])
        x1, y1, x2, y2 = best_box['x1'], best_box['y1'], best_box['x2'], best_box['y2']
        serial_crop = img[y1:y2, x1:x2]

        if serial_crop.size == 0:
            continue

        # Segment into characters
        chars = segment_characters(serial_crop)
        mapping = map_characters_to_serial(chars, bill['serial'])
        if mapping is None:
            continue

        scanned += 1

        # Extract character crops and add to atlas
        for char_val, bound in mapping:
            char_crop = serial_crop[bound['top']:bound['bottom'],
                                    bound['x1']:bound['x2']]
            if char_crop.size == 0:
                continue

            if char_val not in atlas:
                atlas[char_val] = []
            atlas[char_val].append(char_crop)

    print(f"  Scanned {scanned} bills for atlas")
    digit_counts = {d: len(atlas.get(d, [])) for d in '0123456789'}
    print(f"  Digit coverage: {digit_counts}")

    letter_count = sum(1 for k in atlas if k.isalpha())
    print(f"  Letters collected: {letter_count}")

    return atlas


def get_best_donor(atlas, char, target_height):
    """Select the best donor crop for a character, matching target height.

    Args:
        atlas: Character atlas dict.
        char: Target character (e.g., '4').
        target_height: Desired ink height in pixels.

    Returns:
        Crop image (numpy array) or None if character not in atlas.
    """
    crops = atlas.get(char, [])
    if not crops:
        return None

    # Find crop with closest height
    best = min(crops, key=lambda c: abs(c.shape[0] - target_height))
    return best


def composite_serial(img, serial_box, target_serial, atlas):
    """Replace the serial number in a region with a target serial.

    Args:
        img: Full bill image (will be modified in-place).
        serial_box: YOLO detection dict for the serial region.
        target_serial: Target serial string (e.g., "A12344321B").
        atlas: Character atlas dict.

    Returns:
        (success: bool, message: str)
    """
    x1, y1, x2, y2 = serial_box['x1'], serial_box['y1'], serial_box['x2'], serial_box['y2']
    serial_crop = img[y1:y2, x1:x2].copy()

    if serial_crop.size == 0:
        return False, "Empty serial region"

    # Segment existing characters
    chars = segment_characters(serial_crop)
    if len(chars) < 8:
        return False, f"Only segmented {len(chars)} characters (need >= 8)"

    # Determine which character positions contain digits (only replace those).
    # Prefix/suffix letters stay from the base bill — we only composite digits.
    digits = target_serial[1:-1]  # 8 digits
    n = len(chars)
    if n == 10:
        # Full serial: chars[0]=prefix, chars[1:9]=digits, chars[9]=suffix
        digit_pairs = [(digits[i], chars[i + 1]) for i in range(8)]
    elif n == 8:
        # Just digits (prefix/suffix outside YOLO box)
        digit_pairs = [(digits[i], chars[i]) for i in range(8)]
    elif n == 9:
        # One end missing — heuristic: check first char width
        if chars[0]['width'] < chars[1]['width'] * 0.7:
            # Prefix present, suffix missing: chars[0]=prefix, chars[1:9]=digits
            digit_pairs = [(digits[i], chars[i + 1]) for i in range(8)]
        else:
            # Prefix missing, suffix present: chars[0:8]=digits, chars[8]=suffix
            digit_pairs = [(digits[i], chars[i]) for i in range(8)]
    elif n > 10:
        # More than expected — assume first and last are letters
        inner = chars[1:-1]
        if len(inner) >= 8:
            digit_pairs = [(digits[i], inner[i]) for i in range(8)]
        else:
            return False, f"Unexpected character count: {n}"
    else:
        return False, f"Unexpected character count: {n}"

    # Replace each digit character
    missing_chars = []
    for target_char, bound in digit_pairs:
        donor = get_best_donor(atlas, target_char, bound['height'])
        if donor is None:
            missing_chars.append(target_char)
            continue

        # Resize donor to match base character dimensions
        target_w = bound['x2'] - bound['x1']
        target_h = bound['bottom'] - bound['top']
        if target_w <= 0 or target_h <= 0:
            continue

        resized = cv2.resize(donor, (target_w, target_h),
                             interpolation=cv2.INTER_CUBIC)

        # Get the background color from above the character ink area
        bg_region = serial_crop[0:max(1, bound['top']), bound['x1']:bound['x2']]
        if bg_region.size > 0:
            bg_color = np.median(bg_region, axis=(0, 1)).astype(np.uint8)
        else:
            bg_color = np.array([230, 230, 220], dtype=np.uint8)  # Default bill paper

        # Clear the character column (fill with background)
        serial_crop[0:serial_crop.shape[0], bound['x1']:bound['x2']] = bg_color

        # Paste the resized donor at the ink position
        serial_crop[bound['top']:bound['bottom'],
                    bound['x1']:bound['x2']] = resized

    if missing_chars:
        unique_missing = sorted(set(missing_chars))
        return False, f"Missing atlas entries for: {', '.join(unique_missing)}"

    # Write modified crop back to image
    img[y1:y2, x1:x2] = serial_crop
    return True, "OK"


def generate_serial_for_pattern(pattern_name, engine, count=1):
    """Generate serial numbers that match a specific pattern.

    Tries multiple strategies:
    1. Use pattern's Examples field from Lua header
    2. Algorithmic generators for common patterns
    3. Brute-force random generation

    Args:
        pattern_name: Pattern name (e.g., "RADAR").
        engine: PatternEngineV3 instance.
        count: Number of serials to generate.

    Returns:
        List of full serial strings (e.g., ["A12344321B"]).
    """
    results = []
    prefixes = 'ABCDEFGHIJKL'
    suffixes = 'ABCDEFGHIJKLMNOPQRSTUVWXY'

    # Strategy 1: Use Examples from Lua header
    info = engine.lua_patterns.get(pattern_name)
    if info and info.examples:
        for ex in info.examples:
            if len(results) >= count:
                break
            digits = ex.strip()
            if len(digits) == 8 and digits.isdigit():
                prefix = random.choice(prefixes)
                suffix = random.choice(suffixes)
                serial = f"{prefix}{digits}{suffix}"
                # Verify it actually matches
                matches = engine.classify_simple(serial)
                if pattern_name in matches:
                    results.append(serial)

    if len(results) >= count:
        return results[:count]

    # Strategy 2: Algorithmic generators for common patterns
    generated_digits = _algorithmic_generate(pattern_name, count - len(results))
    for digits in generated_digits:
        if len(results) >= count:
            break
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        serial = f"{prefix}{digits}{suffix}"
        matches = engine.classify_simple(serial)
        if pattern_name in matches:
            results.append(serial)

    if len(results) >= count:
        return results[:count]

    # Strategy 3: Brute-force random generation (limited to avoid long waits)
    attempts = 0
    max_attempts = 10000
    while len(results) < count and attempts < max_attempts:
        attempts += 1
        digits = ''.join(str(random.randint(0, 9)) for _ in range(8))
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        serial = f"{prefix}{digits}{suffix}"
        matches = engine.classify_simple(serial)
        if pattern_name in matches:
            # Avoid duplicates
            if serial not in results:
                results.append(serial)

    return results[:count]


def _algorithmic_generate(pattern_name, count):
    """Generate digit strings algorithmically for common patterns.

    Returns list of 8-digit strings (without prefix/suffix).
    """
    results = []

    if pattern_name == 'SOLID':
        for d in range(10):
            results.append(str(d) * 8)
    elif pattern_name == 'NEAR_SOLID':
        for d in range(10):
            for pos in range(8):
                for other in range(10):
                    if other != d:
                        s = list(str(d) * 8)
                        s[pos] = str(other)
                        results.append(''.join(s))
    elif pattern_name == 'RADAR':
        for _ in range(count * 3):
            d = [random.randint(0, 9) for _ in range(4)]
            results.append(''.join(str(x) for x in d + d[::-1]))
    elif pattern_name == 'REPEATER':
        for _ in range(count * 3):
            d = [random.randint(0, 9) for _ in range(4)]
            results.append(''.join(str(x) for x in d + d))
    elif pattern_name == 'SUPER_REPEATER':
        for _ in range(count * 3):
            d = [random.randint(0, 9) for _ in range(2)]
            results.append(''.join(str(x) for x in d * 4))
    elif pattern_name == 'LADDER':
        for start in range(3):  # 01234567, 12345678, 23456789
            results.append(''.join(str(start + i) for i in range(8)))
        for start in range(3):
            results.append(''.join(str(start + 7 - i) for i in range(8)))
    elif pattern_name == 'BOOKENDS':
        for _ in range(count * 3):
            b1, b2 = random.randint(0, 9), random.randint(0, 9)
            mid = [random.randint(0, 9) for _ in range(4)]
            results.append(f"{b1}{b2}{''.join(str(x) for x in mid)}{b1}{b2}")
    elif pattern_name == 'SEVEN_OF_A_KIND':
        for d in range(10):
            for pos in range(8):
                for other in range(10):
                    if other != d:
                        s = [str(d)] * 8
                        s[pos] = str(other)
                        results.append(''.join(s))
    elif pattern_name == 'SEVEN_IN_A_ROW':
        for d in range(10):
            for start in range(2):  # position 0 or 1
                s = [str(d)] * 7
                other = (d + 1) % 10
                if start == 0:
                    results.append(''.join(s) + str(other))
                else:
                    results.append(str(other) + ''.join(s))
    elif pattern_name == 'BINARY':
        for d1 in range(10):
            for d2 in range(d1 + 1, 10):
                for _ in range(3):
                    s = ''.join(random.choice([str(d1), str(d2)]) for _ in range(8))
                    results.append(s)
    elif pattern_name == 'TRINARY':
        for _ in range(count * 5):
            digits = random.sample(range(10), 3)
            s = ''.join(str(random.choice(digits)) for _ in range(8))
            results.append(s)

    random.shuffle(results)
    return results[:count * 3]  # Return extras for verification filtering


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic test bills with specific serial numbers')
    parser.add_argument('--serial', type=str, default=None,
                        help='Specific serial number to generate (e.g., A12344321B)')
    parser.add_argument('--pattern', type=str, default=None,
                        help='Generate serials matching this pattern name')
    parser.add_argument('--all-patterns', action='store_true',
                        help='Generate one test bill for every pattern')
    parser.add_argument('--count', type=int, default=1,
                        help='Number of serials per pattern (default: 1)')
    parser.add_argument('--results', nargs='+', required=True,
                        help='Glob patterns for results CSV files (e.g., archive/*/results.csv)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for generated test images')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report what can be generated without actually generating')
    args = parser.parse_args()

    if not args.serial and not args.pattern and not args.all_patterns:
        parser.error("Must specify one of --serial, --pattern, or --all-patterns")

    project_root = Path(__file__).parent.parent

    # Load bill inventory
    print("Loading bill inventory...")
    bills = load_inventory(args.results)
    if not bills:
        print("Error: No bills found in results CSVs. Check your --results paths.")
        sys.exit(1)
    print(f"  Found {len(bills)} bills")

    # Initialize pattern engine for serial generation/verification
    from pattern_engine_v3 import PatternEngineV3
    engine = PatternEngineV3()

    # Determine target serials
    targets = []  # List of (serial, pattern_name, source_desc)

    if args.serial:
        serial = args.serial.upper()
        if len(serial) != 10:
            print(f"Error: Serial must be 10 characters (letter + 8 digits + letter), got {len(serial)}")
            sys.exit(1)
        matches = engine.classify_simple(serial)
        pattern_label = ', '.join(matches) if matches else 'NONE'
        targets.append((serial, pattern_label, f"user-specified"))

    elif args.pattern:
        pattern_name = args.pattern.upper()
        if pattern_name in SKIP_PATTERNS:
            print(f"Error: {pattern_name} cannot be tested via digit compositing "
                  f"(reason: {_skip_reason(pattern_name)})")
            sys.exit(1)
        if pattern_name not in engine.lua_patterns:
            print(f"Error: Unknown pattern '{pattern_name}'")
            available = sorted(engine.lua_patterns.keys())
            print(f"Available: {', '.join(available[:20])}...")
            sys.exit(1)

        print(f"Generating {args.count} serial(s) for {pattern_name}...")
        serials = generate_serial_for_pattern(pattern_name, engine, args.count)
        if not serials:
            print(f"  Failed to generate any serials matching {pattern_name}")
            sys.exit(1)
        for s in serials:
            targets.append((s, pattern_name, "generated"))

    elif args.all_patterns:
        all_names = sorted(engine.lua_patterns.keys())
        eligible = [n for n in all_names
                    if n not in SKIP_PATTERNS and engine.lua_patterns[n].enabled]
        print(f"Generating serials for {len(eligible)} patterns (count={args.count} each)...")
        skipped = []
        failed = []
        for idx, name in enumerate(all_names):
            if name in SKIP_PATTERNS:
                skipped.append(name)
                continue
            info = engine.lua_patterns[name]
            if not info.enabled:
                skipped.append(name)
                continue
            print(f"  [{idx+1}/{len(all_names)}] {name}...", end='', flush=True)
            serials = generate_serial_for_pattern(name, engine, args.count)
            if serials:
                for s in serials:
                    targets.append((s, name, "generated"))
                print(f" {len(serials)} serial(s)")
            else:
                failed.append(name)
                print(" FAILED")

        if skipped:
            print(f"  Skipped {len(skipped)} patterns: {', '.join(skipped)}")
        if failed:
            print(f"  Failed to generate serials for {len(failed)} patterns: {', '.join(failed)}")

    print(f"\n{len(targets)} target serial(s) to generate")

    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN — Target Serials")
        print(f"{'='*60}")
        for serial, pattern, source in targets:
            # Verify with engine
            matches = engine.classify_simple(serial)
            match_str = ', '.join(matches) if matches else 'NONE'
            status = "OK" if pattern.split(', ')[0] in matches else "MISMATCH"
            print(f"  [{status}] {serial} -> {match_str} ({source})")
        print(f"\nTotal: {len(targets)} serials ready for generation")
        print("Re-run without --dry-run and with --output-dir to generate images.")
        return

    if args.output_dir is None:
        print("\nUse --output-dir to generate composite test images, or --dry-run to preview.")
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

    # Build digit atlas
    print("\nBuilding digit atlas...")
    atlas = build_digit_atlas(bills, model)

    # Check atlas has all digits
    missing_digits = [d for d in '0123456789' if d not in atlas]
    if missing_digits:
        print(f"Warning: Atlas missing digits: {', '.join(missing_digits)}")
        print("Some serials may fail to generate.")

    # Generate composites
    print(f"\nGenerating {len(targets)} composite bill(s)...")
    yolo_cache = {}
    generated = 0
    failed = 0
    report_lines = []

    # Pick a pool of base bills (ones with good YOLO detection)
    base_pool = _build_base_pool(bills, model, yolo_cache, max_size=10)
    if not base_pool:
        print("Error: Could not find any bills with usable YOLO serial detection.")
        sys.exit(1)
    print(f"  Base pool: {len(base_pool)} bills with good serial detection\n")

    for i, (target_serial, pattern_name, source) in enumerate(targets):
        # Create output subdirectory
        safe_pattern = pattern_name.replace(', ', '_').replace(' ', '_')
        dir_name = f"{safe_pattern}_{i+1:03d}"
        out_dir = output_dir / dir_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Pick a random base bill from the pool
        base_entry = random.choice(base_pool)
        base_bill = base_entry['bill']
        base_img = base_entry['img'].copy()
        base_dets = base_entry['detections']

        # Find all serial_number boxes
        serial_boxes = [d for d in base_dets if d['class'] == 'serial_number']
        if not serial_boxes:
            report_lines.append(f"FAIL {dir_name} — No serial boxes detected")
            failed += 1
            continue

        # Composite each serial region
        success = True
        fail_msg = ""
        for box in serial_boxes:
            ok, msg = composite_serial(base_img, box, target_serial, atlas)
            if not ok:
                success = False
                fail_msg = msg
                break

        if not success:
            report_lines.append(f"FAIL {dir_name} {target_serial} — {fail_msg}")
            print(f"  FAIL {dir_name} {target_serial} — {fail_msg}")
            failed += 1
            continue

        # Save composite front image
        front_path = out_dir / 'front.jpg'
        cv2.imwrite(str(front_path), base_img)

        # Copy back image with _b suffix for auto-pairing
        back_path = out_dir / 'front_b.jpg'
        if base_bill['back_file'] and Path(base_bill['back_file']).exists():
            shutil.copy2(base_bill['back_file'], back_path)
        else:
            # Create placeholder
            placeholder = np.ones_like(base_img) * 240
            cv2.putText(placeholder, "No back image", (50, placeholder.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
            cv2.imwrite(str(back_path), placeholder)

        # Write recipe file
        recipe_path = out_dir / 'recipe.txt'
        with open(recipe_path, 'w') as f:
            f.write(f"Target serial: {target_serial}\n")
            f.write(f"Expected pattern: {pattern_name}\n")
            f.write(f"Source: {source}\n")
            f.write(f"\nBase bill: {base_bill['serial']}\n")
            f.write(f"  Front: {base_bill['front_file']}\n")
            f.write(f"  Alignment: angle={base_bill['align_angle']}, "
                    f"flipped={base_bill['align_flipped']}\n")
            f.write(f"\nSerial regions composited: {len(serial_boxes)}\n")
            f.write(f"Atlas characters used: {', '.join(sorted(set(target_serial)))}\n")

        generated += 1
        report_lines.append(f"OK   {dir_name} {target_serial} -> {pattern_name}")
        print(f"  OK   {dir_name} {target_serial} -> {pattern_name}")

    # Write summary report
    report_path = output_dir / 'report.txt'
    with open(report_path, 'w') as f:
        f.write("SERIAL COMPOSITING TEST BILL GENERATION REPORT\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Source: {len(bills)} bills from results CSVs\n")
        f.write(f"Atlas: {sum(len(v) for v in atlas.values())} character crops\n")
        f.write(f"Targets: {len(targets)}\n")
        f.write(f"Generated: {generated}\n")
        f.write(f"Failed: {failed}\n\n")
        for line in report_lines:
            f.write(f"{line}\n")

    print(f"\nGenerated {generated} of {len(targets)} test bills")
    if failed:
        print(f"({failed} failed — see {report_path})")
    print(f"Output: {output_dir}/")
    print(f"Report: {report_path}")


def _build_base_pool(bills, model, yolo_cache, max_size=10):
    """Find bills with good YOLO serial detection to use as base images.

    Returns list of dicts with 'bill', 'img', 'detections' keys.
    """
    pool = []
    sample = list(bills)
    random.shuffle(sample)

    for bill in sample:
        if len(pool) >= max_size:
            break

        key = bill['front_file']
        if key in yolo_cache:
            img, dets = yolo_cache[key]
        else:
            img = apply_cached_alignment(bill['front_file'],
                                         bill['align_angle'],
                                         bill['align_flipped'])
            if img is None:
                continue
            dets = detect_regions(model, img)
            yolo_cache[key] = (img, dets)

        serial_boxes = [d for d in dets if d['class'] == 'serial_number']
        if len(serial_boxes) >= 2:
            # Good candidate — has both serial regions
            pool.append({
                'bill': bill,
                'img': img,
                'detections': dets,
            })

    return pool


def _skip_reason(pattern_name):
    """Return human-readable reason why a pattern is skipped."""
    reasons = {
        'GAS_PUMP': 'requires physical misalignment measurement',
        'STAR': 'requires star symbol image',
        'LOW_RUNS': 'requires metadata (series/district/block)',
        'KNOWN_SERIALS': 'requires external data file match',
    }
    return reasons.get(pattern_name, 'unknown')


if __name__ == '__main__':
    main()
