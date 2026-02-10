#!/usr/bin/env python3
"""Debug script to visualize pairwise median seal shift detection."""

import cv2
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from process_production import ProductionProcessor

# Initialize processor once
script_dir = Path(__file__).parent.parent
model_path = script_dir / "best.pt"
processor = ProductionProcessor(model_path)

# Expected pair distances calibrated from 10 reference bills in canon/
# Format: (overprint_class_id, intaglio_class_id): (expected_dx, expected_dy)
EXPECTED_PAIR_DISTANCES = {
    (5, 3): (16.1788, 23.5241),    # seal_f vs denomination
    (5, 4): (-61.1580, -18.3492),  # seal_f vs front_plate
    (5, 8): (-40.7820, -28.4867),  # seal_f vs series_year
    (6, 3): (64.4764, 31.6387),    # seal_t vs denomination
    (6, 4): (-12.6970, -10.3124),  # seal_t vs front_plate
    (6, 8): (7.5330, -20.5299),    # seal_t vs series_year
    (7, 3): (39.3343, 23.4173),    # serial_number vs denomination
    (7, 4): (-37.8311, -18.5696),  # serial_number vs front_plate
    (7, 8): (-17.6118, -28.4774),  # serial_number vs series_year
}

# Class IDs and names
CLASS_NAME_TO_ID = {
    'seal_f': 5, 'seal_t': 6, 'serial_number': 7,
    'denomination': 3, 'front_plate': 4, 'series_year': 8
}
CLASS_ID_TO_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}

CONF_THRESHOLDS = {5: 0.3, 6: 0.3, 7: 0.5, 3: 0.3, 4: 0.3, 8: 0.3}
OVERPRINT_IDS = {5, 6, 7}   # seal_f, seal_t, serial_number
INTAGLIO_IDS = {3, 4, 8}    # denomination, front_plate, series_year


def debug_overprint_shift(image_path: str):
    """Show pairwise median shift calculation with visual output."""
    global processor

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read: {image_path}")
        return

    h, w = img.shape[:2]
    print(f"\n{'='*60}")
    print(f"{Path(image_path).name}: {w}x{h}")
    print(f"{'='*60}")

    # Use lower conf to get more detections (same as _calculate_seal_shift)
    detections = processor._detect_all_objects(img, conf=0.1)

    # Collect boxes by class ID, filter by confidence
    by_class = {}
    for name, class_id in CLASS_NAME_TO_ID.items():
        positions = []
        boxes = []
        for box in detections.get(name, []):
            if box[4] >= CONF_THRESHOLDS[class_id]:
                cx = (box[0] + box[2]) / 2 / w * 100
                cy = (box[1] + box[3]) / 2 / h * 100
                positions.append((cx, cy))
                boxes.append(box)
        if positions:
            by_class[class_id] = {'positions': positions, 'boxes': boxes}

    # Compute centroid for each class
    class_positions = {}
    for cls_id, data in by_class.items():
        positions = data['positions']
        class_positions[cls_id] = (
            sum(p[0] for p in positions) / len(positions),
            sum(p[1] for p in positions) / len(positions),
        )

    # Print per-class centroids
    print(f"\nPER-CLASS CENTROIDS:")
    print(f"  OVERPRINT (letterpress):")
    for cls_id in sorted(OVERPRINT_IDS):
        if cls_id in class_positions:
            cx, cy = class_positions[cls_id]
            n = len(by_class[cls_id]['positions'])
            print(f"    {CLASS_ID_TO_NAME[cls_id]:15s}: ({cx:6.2f}%, {cy:6.2f}%)  [{n} detection(s)]")
        else:
            print(f"    {CLASS_ID_TO_NAME[cls_id]:15s}: NOT DETECTED")

    print(f"  INTAGLIO (face plate):")
    for cls_id in sorted(INTAGLIO_IDS):
        if cls_id in class_positions:
            cx, cy = class_positions[cls_id]
            n = len(by_class[cls_id]['positions'])
            print(f"    {CLASS_ID_TO_NAME[cls_id]:15s}: ({cx:6.2f}%, {cy:6.2f}%)  [{n} detection(s)]")
        else:
            print(f"    {CLASS_ID_TO_NAME[cls_id]:15s}: NOT DETECTED")

    # Compute deviation from expected for each pair
    print(f"\nPAIRWISE DEVIATIONS (actual - expected):")
    pair_deviations = []
    for (o_cls, i_cls), (exp_dx, exp_dy) in sorted(EXPECTED_PAIR_DISTANCES.items()):
        o_name = CLASS_ID_TO_NAME[o_cls]
        i_name = CLASS_ID_TO_NAME[i_cls]
        pair_label = f"{o_name} vs {i_name}"

        if o_cls in class_positions and i_cls in class_positions:
            ox, oy = class_positions[o_cls]
            ix, iy = class_positions[i_cls]
            actual_dx = ox - ix
            actual_dy = oy - iy
            dev_x = actual_dx - exp_dx
            dev_y = actual_dy - exp_dy
            pair_deviations.append((dev_x, dev_y))
            print(f"  {pair_label:30s}: dX={dev_x:+6.3f}%  dY={dev_y:+6.3f}%")
        else:
            print(f"  {pair_label:30s}: MISSING DETECTION")

    if not pair_deviations:
        print(f"\n*** Insufficient detections for shift calculation ***")
        return

    # Median deviation = robust shift estimate
    shift_x = statistics.median([d[0] for d in pair_deviations])
    shift_y = statistics.median([d[1] for d in pair_deviations])

    print(f"\nMEDIAN SHIFT (from {len(pair_deviations)} pairs):")
    print(f"  X: {shift_x:+.3f}%")
    print(f"  Y: {shift_y:+.3f}%")

    # Flag if deviation exceeds thresholds
    flags = []
    if shift_y < -1.7:
        flags.append("HIGH_SEAL")
    if shift_y > 1.3:
        flags.append("LOW_SEAL")
    if abs(shift_y) > 1.5:
        flags.append("SEAL_SHIFT")

    if flags:
        print(f"\n*** FLAGS: {', '.join(flags)} ***")
    else:
        print(f"\n(No shift flags - within normal range)")

    # Draw visualization
    # Overprint boxes in RED, Intaglio boxes in GREEN
    for cls_id in OVERPRINT_IDS:
        if cls_id in by_class:
            for box in by_class[cls_id]['boxes']:
                x1, y1, x2, y2, conf = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(img, f"{CLASS_ID_TO_NAME[cls_id]} {conf:.2f}",
                           (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

    for cls_id in INTAGLIO_IDS:
        if cls_id in by_class:
            for box in by_class[cls_id]['boxes']:
                x1, y1, x2, y2, conf = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f"{CLASS_ID_TO_NAME[cls_id]} {conf:.2f}",
                           (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

    # Save debug image
    out_path = Path(image_path).stem + "_debug.jpg"
    cv2.imwrite(out_path, img)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Process command line arguments
        for arg in sys.argv[1:]:
            path = Path(arg)
            if path.is_dir():
                # Process all images in directory
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    for f in sorted(path.glob(ext)):
                        debug_overprint_shift(str(f))
            elif path.exists():
                debug_overprint_shift(str(path))
            else:
                print(f"Not found: {arg}")
    else:
        # Default test files
        test_files = [
            "/home/pbarros/Pictures/Dollar/seal_test/seal-1.jpg",
            "/home/pbarros/Pictures/Dollar/seal_test/seal-10.jpg",
            "/home/pbarros/Pictures/Dollar/seal_test/seal-11.jpg",
        ]
        for f in test_files:
            if Path(f).exists():
                debug_overprint_shift(f)
            else:
                print(f"Test file not found: {f}")
        print("\nUsage: python tools/debug_seal_detection.py <image_or_directory> ...")
