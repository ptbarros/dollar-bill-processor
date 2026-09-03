"""Validate RapidOCR vs EasyOCR through the FULL serial pipeline, against a
results CSV as reference (the serials the app already produced with EasyOCR).

Runs extract_serial() (incl. the treasury-seal Fed-letter cross-check) both
ways on each front image and reports:
  * EasyOCR vs CSV   (sanity; should be ~100% since the CSV came from EasyOCR)
  * RapidOCR vs CSV  (the real accuracy measure)
  * RapidOCR vs EasyOCR agreement
  * every RapidOCR mismatch, flagging first-letter-only differences

Usage:
  venv/bin/python tools/ocr_validate.py --csv ~/Pictures/829/results_*.csv [--limit N]
"""

import argparse
import csv
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from process_production import ProductionProcessor
from ocr_backend import EasyOCRBackend, RapidOCRBackend


def load_refs(csv_glob):
    path = sorted(glob.glob(str(Path(csv_glob).expanduser())))[-1]
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            front = r.get("front_file", "").strip()
            serial = (r.get("serial") or "").strip()
            if front and serial:
                rows.append((front, serial))
    return path, rows


def safe_serial(proc, path):
    try:
        return (proc.extract_serial(Path(path))[0] or "").strip()
    except Exception as e:
        return f"<err:{type(e).__name__}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    csv_path, refs = load_refs(args.csv)
    if args.limit:
        refs = refs[: args.limit]
    print(f"Reference CSV: {csv_path}")
    print(f"Validating {len(refs)} bills through the full pipeline\n", flush=True)

    proc = ProductionProcessor("best.pt")
    easy = EasyOCRBackend(use_gpu=False)
    rapid = RapidOCRBackend(use_gpu=False)

    easy_ok = rapid_ok = agree = 0
    rapid_misses = []
    for i, (front, ref) in enumerate(refs, 1):
        proc.ocr_reader = easy
        es = safe_serial(proc, front)
        proc.ocr_reader = rapid
        rs = safe_serial(proc, front)

        easy_ok += (es == ref)
        rapid_ok += (rs == ref)
        agree += (es == rs)
        if rs != ref:
            first_only = (len(rs) == len(ref) and rs[1:] == ref[1:] and rs[:1] != ref[:1])
            rapid_misses.append((Path(front).name, ref, es, rs, first_only))
        if i % 10 == 0:
            print(f"  ...{i}/{len(refs)}", flush=True)

    n = len(refs) or 1
    print("\n" + "=" * 60)
    print(f"EasyOCR  vs CSV reference : {easy_ok}/{len(refs)} ({100*easy_ok/n:.0f}%)")
    print(f"RapidOCR vs CSV reference : {rapid_ok}/{len(refs)} ({100*rapid_ok/n:.0f}%)")
    print(f"RapidOCR agrees w/ EasyOCR: {agree}/{len(refs)} ({100*agree/n:.0f}%)")
    print("=" * 60)

    if rapid_misses:
        fl = sum(1 for m in rapid_misses if m[4])
        print(f"\nRapidOCR misses vs reference: {len(rapid_misses)} "
              f"({fl} are first-letter-only)")
        for name, ref, es, rs, first_only in rapid_misses:
            tag = "  [FIRST-LETTER]" if first_only else ""
            print(f"  {name}: ref={ref}  easy={es}  rapid={rs}{tag}")


if __name__ == "__main__":
    main()
