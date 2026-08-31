"""Validate RapidOCR vs EasyOCR on the SMALL-FONT fields: series year, front
plate, back plate — the heavily hand-tuned area (fw_corrections / letter
corrections). Reference is a results CSV produced with EasyOCR.

Replicates processing_thread.py: align front/back with yolo_aligner, then call
_extract_plate_info() with each OCR backend.

Usage:
  venv/bin/python tools/ocr_validate_plates.py --csv ~/Pictures/829/results_*.csv [--limit N]
"""

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from process_production import ProductionProcessor
from ocr_backend import EasyOCRBackend, RapidOCRBackend


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().upper())


def load_refs(csv_glob):
    path = sorted(glob.glob(str(Path(csv_glob).expanduser())))[-1]
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return path, rows


def plate_read(proc, front, back):
    try:
        fa, _ = proc.yolo_aligner.align_image(Path(front))
        ba = None
        if back:
            ba, _ = proc.yolo_aligner.align_image(Path(back))
        info = proc._extract_plate_info(fa, ba)
        return info.get("series_year", ""), info.get("front_plate", ""), info.get("back_plate", "")
    except Exception as e:
        return (f"<err:{type(e).__name__}>",) * 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    csv_path, refs = load_refs(args.csv)
    if args.limit:
        refs = refs[: args.limit]
    print(f"Reference CSV: {csv_path}")
    print(f"Validating plate/series on {len(refs)} bills\n", flush=True)

    proc = ProductionProcessor("best.pt")
    easy = EasyOCRBackend(use_gpu=False)
    rapid = RapidOCRBackend(use_gpu=False)

    fields = ["series_year", "front_plate", "back_plate"]
    easy_ok = {f: 0 for f in fields}
    rapid_ok = {f: 0 for f in fields}
    agree = {f: 0 for f in fields}
    front_misses = []

    for i, r in enumerate(refs, 1):
        front, back = r["front_file"], r.get("back_file", "")
        ref = {f: norm(r.get(f, "")) for f in fields}

        proc.ocr_reader = easy
        ey, ef, eb = map(norm, plate_read(proc, front, back))
        proc.ocr_reader = rapid
        ry, rf, rb = map(norm, plate_read(proc, front, back))
        e = {"series_year": ey, "front_plate": ef, "back_plate": eb}
        rp = {"series_year": ry, "front_plate": rf, "back_plate": rb}

        for f in fields:
            easy_ok[f] += (e[f] == ref[f])
            rapid_ok[f] += (rp[f] == ref[f])
            agree[f] += (e[f] == rp[f])
        if rp["front_plate"] != ref["front_plate"]:
            front_misses.append((Path(front).name, ref["front_plate"], e["front_plate"], rp["front_plate"]))
        if i % 10 == 0:
            print(f"  ...{i}/{len(refs)}", flush=True)

    n = len(refs) or 1
    print("\n" + "=" * 64)
    print(f"{'field':14s} {'EasyOCR vs CSV':>16s} {'RapidOCR vs CSV':>17s} {'agree':>8s}")
    for f in fields:
        print(f"{f:14s} {easy_ok[f]:>7d}/{len(refs):<7d} {rapid_ok[f]:>8d}/{len(refs):<7d} "
              f"{100*agree[f]/n:>6.0f}%")
    print("=" * 64)

    if front_misses:
        print(f"\nfront_plate mismatches vs reference: {len(front_misses)}")
        for name, ref, es, rs in front_misses:
            flag = "  <-- rapid-only" if es == ref else ("  (easy also off)" if es != ref else "")
            print(f"  {name}: ref={ref!r}  easy={es!r}  rapid={rs!r}{flag}")


if __name__ == "__main__":
    main()
