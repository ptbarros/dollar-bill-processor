#!/usr/bin/env python3
"""Vertical-feed rotation-recovery test harness.

Draft companion to the vertical-feed recovery added in process_production.py
(classify_and_cache_image + rotate_coarse). Use it to validate the recovery on
real vertical scans BEFORE trusting it in the GUI pipeline:

    venv/bin/python tools/test_vertical_feed.py <folder-or-image> [more...]
    venv/bin/python tools/test_vertical_feed.py <folder> --write /tmp/derotated

For every image it runs the detector at 0/90/180/270 degrees and prints each
orientation's front/back confidence, serial-box count, and the recovery score,
then the rotation the recovery WOULD pick (the one classify_and_cache_image
returns as ``coarse_rotation``). Fronts and backs are scored independently, so a
vertically-fed pair should show OPPOSITE winning turns -- exactly the case a
folder-wide rotate can't handle.

With --write, the chosen landscape image is saved next to a copy of the original
so you can eyeball that serials ended up horizontal. Nothing in the source
folder is modified.

Expected healthy result (from the empirical baseline in project memory):
  - a correctly landscape front:  chosen=0,   front_conf ~0.8+, 2 serial boxes
  - a front fed vertically:        chosen=90 or 270, that angle ~0.8+ / 2 serials
  - its back (flipped to scan):    chosen = the OPPOSITE 90-deg turn
A sideways orientation reads ~0.42-0.46 conf with a garbage serial count.
"""
import sys
from pathlib import Path

# Allow running from anywhere: put the project root on sys.path.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
from process_production import (  # noqa: E402
    ProductionProcessor, Config, rotate_coarse,
)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def _resolve_model():
    """Same resolution the app uses: obscured detector.bin if present, else
    best.onnx / best.pt from the project root."""
    obf = ROOT / "detector.bin"
    if obf.exists():
        return obf, obf
    onnx = ROOT / "best.onnx"
    return ROOT / "best.pt", (onnx if onnx.exists() else None)


def _gather(args):
    paths = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            paths += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            paths.append(p)
        else:
            print(f"  (skipping {a!r}: not an image or folder)")
    return paths


def main():
    argv = [a for a in sys.argv[1:] if a != '--write']
    write_dir = None
    if '--write' in sys.argv:
        i = sys.argv.index('--write')
        if i + 1 < len(sys.argv):
            write_dir = Path(sys.argv[i + 1])
            argv = [a for a in argv if a != str(write_dir)]
            write_dir.mkdir(parents=True, exist_ok=True)

    if not argv:
        print(__doc__)
        return 1

    images = _gather(argv)
    if not images:
        print("No images found.")
        return 1

    yolo_path, onnx_path = _resolve_model()
    if not ((onnx_path and onnx_path.exists()) or yolo_path.exists()):
        print("Detection model not found (looked for detector.bin / best.onnx / best.pt).")
        return 1

    cfg = Config()
    patterns_dir = ROOT / "patterns"
    print(f"Loading model {onnx_path or yolo_path} ...\n")
    proc = ProductionProcessor(
        yolo_path, use_gpu=False, cfg=cfg,
        patterns_dir=patterns_dir if patterns_dir.exists() else None,
        onnx_model_path=onnx_path,
    )

    print(f"{'image':32} {'aspect':8} {'0':>18} {'90':>18} {'180':>18} {'270':>18}  chosen")
    print("-" * 130)

    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            print(f"{path.name:32} (failed to load)")
            continue
        h, w = img.shape[:2]
        aspect = "portrait" if h > w * 1.15 else "landscape"

        cells = []
        for deg in (0, 90, 180, 270):
            d = proc._classify_array(rotate_coarse(img, deg))
            score = proc._orientation_score(d)
            tag = 'F' if d.get('is_front') else 'B'
            cells.append(
                f"{tag} c{max(d.get('front_conf', 0), d.get('back_conf', 0)):.2f}"
                f" s{len(d.get('serial_boxes', []))} ={score:+.2f}"
            )

        # What the real pipeline would decide (includes the gate).
        chosen = proc.classify_and_cache_image(path).get('coarse_rotation', 0)

        print(f"{path.name:32} {aspect:8} " + " ".join(f"{c:>18}" for c in cells)
              + f"  -> {chosen}")

        if write_dir is not None:
            out = write_dir / f"{path.stem}__derot{chosen}{path.suffix}"
            cv2.imwrite(str(out), rotate_coarse(img, chosen),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

    print("\nLegend: F/B = classified front/back, c = max bill conf, "
          "s = serial-box count, = recovery score. 'chosen' is the clockwise "
          "turn the recovery applies (0/90/180/270).")
    if write_dir is not None:
        print(f"De-rotated copies written to {write_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
