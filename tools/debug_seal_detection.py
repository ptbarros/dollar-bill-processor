#!/usr/bin/env python3
"""Debug script to visualize seal vs ONE_hashed shift detection."""

import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from process_production import ProductionProcessor

# Initialize processor once
script_dir = Path(__file__).parent.parent
model_path = script_dir / "best.pt"
processor = ProductionProcessor(model_path)


def debug_seal_shift(image_path: str):
    """Show seal vs ONE_hashed shift calculation with visual output."""
    global processor

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read: {image_path}")
        return

    h, w = img.shape[:2]
    print(f"\n{'='*60}")
    print(f"{Path(image_path).name}: {w}x{h}")
    print(f"{'='*60}")

    # Get detections (same as _calculate_seal_shift)
    detections = processor._detect_all_objects(img, conf=0.1)

    # Get seal_t and ONE_hashed boxes
    seal_boxes = [b for b in detections.get('seal_t', []) if b[4] >= 0.3]
    one_boxes = [b for b in detections.get('ONE_hashed', []) if b[4] >= 0.3]

    print(f"\nDETECTIONS:")
    print(f"  seal_t:      {len(seal_boxes)} box(es)")
    for i, b in enumerate(seal_boxes):
        print(f"    [{i}] conf={b[4]:.3f}  box=({int(b[0])}, {int(b[1])}) - ({int(b[2])}, {int(b[3])})")
    print(f"  ONE_hashed:  {len(one_boxes)} box(es)")
    for i, b in enumerate(one_boxes):
        print(f"    [{i}] conf={b[4]:.3f}  box=({int(b[0])}, {int(b[1])}) - ({int(b[2])}, {int(b[3])})")

    if not seal_boxes or not one_boxes:
        print(f"\n*** Missing detections - cannot calculate shift ***")
        return

    # Use highest confidence detection for each
    seal = max(seal_boxes, key=lambda b: b[4])
    one = max(one_boxes, key=lambda b: b[4])

    # ONE_hashed dimensions
    one_w = one[2] - one[0]
    one_h = one[3] - one[1]

    # Centers
    one_cx = (one[0] + one[2]) / 2
    one_cy = (one[1] + one[3]) / 2
    seal_cx = (seal[0] + seal[2]) / 2
    seal_cy = (seal[1] + seal[3]) / 2

    print(f"\nBOX ANALYSIS:")
    print(f"  ONE_hashed:  center=({one_cx:.1f}, {one_cy:.1f})  size=({one_w:.1f} x {one_h:.1f})")
    print(f"  seal_t:      center=({seal_cx:.1f}, {seal_cy:.1f})")

    # Center-to-center offset as % of ONE dimensions
    # Standard coordinates: +x is right, +y is UP (negate image y)
    dx_pct = (seal_cx - one_cx) / one_w * 100
    dy_pct = -(seal_cy - one_cy) / one_h * 100  # Negate so +y = up, -y = down

    print(f"\nCENTER-TO-CENTER OFFSET:")
    print(f"  dX: {dx_pct:+.2f}% of ONE width")
    print(f"  dY: {dy_pct:+.2f}% of ONE height")

    # Containment calculation
    inter_x1 = max(seal[0], one[0])
    inter_y1 = max(seal[1], one[1])
    inter_x2 = min(seal[2], one[2])
    inter_y2 = min(seal[3], one[3])

    seal_area = (seal[2] - seal[0]) * (seal[3] - seal[1])
    if inter_x2 > inter_x1 and inter_y2 > inter_y1 and seal_area > 0:
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        containment = inter_area / seal_area * 100
    else:
        inter_area = 0
        containment = 0.0

    print(f"\nCONTAINMENT:")
    print(f"  Seal area:         {seal_area:.0f} px²")
    print(f"  Intersection area: {inter_area:.0f} px²")
    print(f"  Containment:       {containment:.1f}%")

    # Overflow per side
    print(f"\nOVERFLOW (seal extending beyond ONE bbox):")
    overflow_left = max(0, one[0] - seal[0])
    overflow_right = max(0, seal[2] - one[2])
    overflow_top = max(0, one[1] - seal[1])
    overflow_bottom = max(0, seal[3] - one[3])
    print(f"  Left:   {overflow_left:.1f} px")
    print(f"  Right:  {overflow_right:.1f} px")
    print(f"  Top:    {overflow_top:.1f} px")
    print(f"  Bottom: {overflow_bottom:.1f} px")

    # Flag if containment below threshold (single threshold approach)
    flags = []
    if containment < 97:
        flags.append("SEAL_SHIFT")

    if flags:
        print(f"\n*** FLAGS: {', '.join(flags)} ***")
    else:
        print(f"\n(No shift flags - within normal range)")

    # Draw visualization
    # ONE_hashed box in GREEN
    cv2.rectangle(img, (int(one[0]), int(one[1])), (int(one[2]), int(one[3])), (0, 255, 0), 2)
    cv2.putText(img, f"ONE_hashed {one[4]:.2f}",
               (int(one[0]), int(one[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.circle(img, (int(one_cx), int(one_cy)), 5, (0, 255, 0), -1)

    # seal_t box in RED
    cv2.rectangle(img, (int(seal[0]), int(seal[1])), (int(seal[2]), int(seal[3])), (0, 0, 255), 2)
    cv2.putText(img, f"seal_t {seal[4]:.2f}",
               (int(seal[0]), int(seal[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.circle(img, (int(seal_cx), int(seal_cy)), 5, (0, 0, 255), -1)

    # Draw line between centers
    cv2.line(img, (int(one_cx), int(one_cy)), (int(seal_cx), int(seal_cy)), (255, 255, 0), 2)

    # Add text overlay with results
    y_pos = 30
    cv2.putText(img, f"dX: {dx_pct:+.2f}%  dY: {dy_pct:+.2f}%", (10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_pos += 25
    cv2.putText(img, f"Containment: {containment:.1f}%", (10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if flags:
        y_pos += 25
        cv2.putText(img, f"FLAGS: {', '.join(flags)}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
                        debug_seal_shift(str(f))
            elif path.exists():
                debug_seal_shift(str(path))
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
                debug_seal_shift(f)
            else:
                print(f"Test file not found: {f}")
        print("\nUsage: python tools/debug_seal_detection.py <image_or_directory> ...")
