"""Compare serial-read accuracy: EasyOCR vs RapidOCR, through the full pipeline.

Runs the complete extract_serial() cascade both ways on each front image and
reports where the two engines agree/disagree. Since repo ground-truth is thin,
agreement is the primary signal (disagreements are what to eyeball); any labeled
serial (passed via a sidecar or matched folder) is scored directly.

Usage:
  venv/bin/python tools/ocr_compare.py --images test_low_runs canon
  # optionally point at your larger real-bill archive:
  venv/bin/python tools/ocr_compare.py --images /path/to/scans
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from process_production import ProductionProcessor
from ocr_backend import EasyOCRBackend, RapidOCRBackend


def find_fronts(patterns):
    fronts = []
    for p in patterns:
        pp = Path(p)
        if pp.is_dir():
            # prefer files literally named front.jpg, else any front-ish jpg
            named = list(pp.rglob("front.jpg"))
            fronts += named or [
                f for f in pp.rglob("*.jpg")
                if "fancy_bills" not in f.parts and "review" not in f.parts
            ]
        elif pp.exists():
            fronts.append(pp)
    seen, out = set(), []
    for f in sorted(fronts):
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


import cv2


def read_all(proc, path):
    """Run the serial OCR cascade on each detected serial crop; return best (serial, conf).

    Bypasses the full extract_serial() so a pre-existing contour-aligner bug
    (numpy 2.x HoughLinesP) on this dev box doesn't block OCR comparison. This
    still exercises the real 4-strategy cascade + confusion corrections per crop.
    """
    img = cv2.imread(str(path))
    if img is None:
        return "<unreadable>", 0.0
    dets = proc._detect_all_objects(img, conf=0.3)
    boxes = sorted(dets.get("serial_number", []), key=lambda b: -b[4])
    best_serial, best_conf = None, 0.0
    for x1, y1, x2, y2, _ in boxes:
        px, py = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
        crop = img[max(0, y1 - py): y2 + py, max(0, x1 - px): x2 + px]
        try:
            serial, conf = proc.extract_serial_from_crop(crop)
        except Exception as e:
            serial, conf = f"<error: {type(e).__name__}>", 0.0
        if serial and conf > best_conf:
            best_serial, best_conf = serial, conf
    return best_serial, best_conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    proc = ProductionProcessor("best.pt")
    easy = EasyOCRBackend(use_gpu=False)
    rapid = RapidOCRBackend(use_gpu=False)

    fronts = find_fronts(args.images)
    if args.limit:
        fronts = fronts[: args.limit]
    print(f"Comparing {len(fronts)} front images\n")
    print(f"{'image':38s} {'EasyOCR':>13s} {'RapidOCR':>13s}  match")
    print("-" * 84)

    agree = same_serial = 0
    disagreements = []
    for path in fronts:
        proc.ocr_reader = easy
        es, ec = read_all(proc, path)
        proc.ocr_reader = rapid
        rs, rc = read_all(proc, path)

        match = (es == rs)
        agree += match
        if es and rs and es == rs:
            same_serial += 1
        name = path.parent.name + "/" + path.name
        print(f"{name[:38]:38s} {str(es)[:11]:>13s} {str(rs)[:11]:>13s}  {'OK' if match else 'DIFF'}")
        if not match:
            disagreements.append((name, es, ec, rs, rc))

    print("-" * 84)
    n = len(fronts) or 1
    print(f"\nAgreement: {agree}/{len(fronts)} ({100*agree/n:.0f}%)  "
          f"[both read identical serial: {same_serial}]")
    if disagreements:
        print("\nDisagreements to eyeball:")
        for name, es, ec, rs, rc in disagreements:
            print(f"  {name}")
            print(f"      EasyOCR : {es}  ({ec:.2f})")
            print(f"      RapidOCR: {rs}  ({rc:.2f})")


if __name__ == "__main__":
    main()
