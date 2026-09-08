#!/usr/bin/env python3
"""
Analyze scan-data zips produced by tools/collect_scans.bat.

Aggregates every results_*.csv across all straps to answer:
  - how many bills were processed (and how many are duplicates seen before)
  - per-pattern HIT RATE (how often each fancy pattern appears in circulation)
  - per-pattern KEEP RATE (of the hits, how many he actually cropped = kept)

The "kept" signal comes from the crop filenames (<SERIAL>_NN.jpg) recorded in
each zip's _all_files_manifest.csv, so we never need the image files themselves.

Usage:
    python tools/analyze_scans.py DBP_data_August_*.zip DBP_data_September_*.zip
    python tools/analyze_scans.py /path/to/extracted_folder
    python tools/analyze_scans.py *.zip --out report_dir/

Dependency-free (Python stdlib only).
"""

import argparse
import csv
import re
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

CROP_RE = re.compile(r"^(?P<serial>.+?)_\d{2,}\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def _iter_sources(paths):
    """Yield (label, root_dir) for each input, extracting zips to a temp dir."""
    tmp = tempfile.mkdtemp(prefix="dbp_analyze_")
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"  ! skipping (not found): {p}", file=sys.stderr)
            continue
        if p.is_dir():
            yield p.name, p
        elif p.suffix.lower() == ".zip":
            dest = Path(tmp) / p.stem
            dest.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(p) as zf:
                    # Windows-made zips (Compress-Archive) use '\' separators, which
                    # extractall() does NOT split into folders on Linux/mac. Rewrite
                    # entry names to forward slashes so rglob() sees real subfolders.
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        rel = info.filename.replace("\\", "/")
                        out = dest / rel
                        out.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(info) as src, open(out, "wb") as f:
                            f.write(src.read())
                yield p.stem, dest
            except zipfile.BadZipFile:
                print(f"  ! skipping (bad zip): {p}", file=sys.stderr)
        else:
            print(f"  ! skipping (not a folder or .zip): {p}", file=sys.stderr)


def _split_patterns(fancy_types):
    """'CS-Radar, CS-Repeater' -> ['CS-Radar', 'CS-Repeater'] (ALL/blank -> [])."""
    if not fancy_types:
        return []
    parts = [t.strip() for t in fancy_types.split(",")]
    return [t for t in parts if t and t.upper() != "ALL"]


def _norm_serial(s):
    return (s or "").strip().upper().replace("*", "STAR")


def analyze(paths):
    total_rows = 0
    error_rows = 0
    fancy_rows = 0
    review_rows = 0

    serial_seen = Counter()            # serial -> times processed (dup detection)
    pattern_hits = Counter()           # pattern -> bills that matched it
    pattern_serials = defaultdict(set) # pattern -> set of serials that matched
    kept_serials = set()               # serials that have a crop file (kept)
    per_source = Counter()             # source label -> bill count
    year_counter = Counter()           # series_year -> count (fun/extra)

    for label, root in _iter_sources(paths):
        # --- results CSVs: the per-bill record ---
        # A strap folder re-processed N times has N timestamped results_*.csv;
        # keep only the newest per folder so re-runs don't count bills N times.
        latest_by_folder = {}
        for csv_path in root.rglob("results_*.csv"):
            key = str(csv_path.parent)
            if key not in latest_by_folder or csv_path.name > latest_by_folder[key].name:
                latest_by_folder[key] = csv_path
        for csv_path in sorted(latest_by_folder.values()):
            try:
                with open(csv_path, newline="", encoding="utf-8-sig", errors="replace") as f:
                    for row in csv.DictReader(f):
                        serial = _norm_serial(row.get("serial"))
                        if not serial:
                            continue
                        total_rows += 1
                        per_source[label] += 1
                        serial_seen[serial] += 1
                        if (row.get("error") or "").strip():
                            error_rows += 1
                        if (row.get("needs_review") or "").strip().lower() in ("true", "1", "yes"):
                            review_rows += 1
                        yr = (row.get("series_year") or "").strip()
                        if yr:
                            year_counter[yr] += 1
                        pats = _split_patterns(row.get("fancy_types"))
                        if pats:
                            fancy_rows += 1
                            for pat in pats:
                                pattern_hits[pat] += 1
                                pattern_serials[pat].add(serial)
            except OSError as e:
                print(f"  ! could not read {csv_path}: {e}", file=sys.stderr)

        # --- manifest: crop filenames = kept serials ---
        for man in root.rglob("_all_files_manifest.csv"):
            try:
                with open(man, newline="", encoding="utf-8-sig", errors="replace") as f:
                    for row in csv.DictReader(f):
                        m = CROP_RE.match((row.get("Name") or "").strip())
                        if m:
                            kept_serials.add(_norm_serial(m.group("serial")))
            except OSError:
                pass
        # Fallback: some folders keep crops as loose files, not just in manifest
        for img in root.rglob("*.jpg"):
            m = CROP_RE.match(img.name)
            if m:
                kept_serials.add(_norm_serial(m.group("serial")))

    # per-pattern keep counts (intersection of matched serials and kept serials)
    pattern_kept = {
        pat: len(serials & kept_serials) for pat, serials in pattern_serials.items()
    }

    return {
        "total_rows": total_rows,
        "unique_serials": len(serial_seen),
        "duplicate_bills": sum(c - 1 for c in serial_seen.values() if c > 1),
        "error_rows": error_rows,
        "fancy_rows": fancy_rows,
        "review_rows": review_rows,
        "kept_total": len(kept_serials),
        "pattern_hits": pattern_hits,
        "pattern_kept": pattern_kept,
        "per_source": per_source,
        "year_counter": year_counter,
        "most_seen": serial_seen.most_common(10),
    }


def print_report(r):
    total = r["total_rows"]
    print("\n" + "=" * 64)
    print("  DOLLAR DETECTIVE — SCAN DATA ANALYSIS")
    print("=" * 64)
    if not total:
        print("\nNo results_*.csv rows found. Check the input paths/zips.")
        return
    print(f"\nBills processed (rows) : {total:,}")
    print(f"Unique serials         : {r['unique_serials']:,}")
    print(f"Duplicate re-scans      : {r['duplicate_bills']:,}")
    print(f"Fancy hits             : {r['fancy_rows']:,}  ({r['fancy_rows']/total:.2%})")
    print(f"Flagged needs-review   : {r['review_rows']:,}")
    print(f"Processing errors      : {r['error_rows']:,}")
    print(f"Bills cropped (kept)   : {r['kept_total']:,}")

    if len(r["per_source"]) > 1:
        print("\nBy source:")
        for label, n in r["per_source"].most_common():
            print(f"  {n:>7,}  {label}")

    print("\n" + "-" * 64)
    print(f"  {'PATTERN':<34}{'HITS':>7}{'RATE':>9}{'KEPT':>6}{'KEEP%':>7}")
    print("-" * 64)
    rows = []
    for pat, hits in r["pattern_hits"].most_common():
        kept = r["pattern_kept"].get(pat, 0)
        rows.append((pat, hits, hits / total, kept, (kept / hits if hits else 0)))
    for pat, hits, rate, kept, keeprate in rows:
        print(f"  {pat[:34]:<34}{hits:>7,}{rate:>8.3%}{kept:>6}{keeprate:>7.1%}")

    if r["most_seen"] and r["most_seen"][0][1] > 1:
        print("\nMost re-scanned serials (seen-before check):")
        for serial, n in r["most_seen"]:
            if n > 1:
                print(f"  {n}x  {serial}")
    return rows


def write_csv(rows, total, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pattern_hit_keep_rates.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pattern", "hits", "hit_rate", "kept", "keep_rate", "total_bills"])
        for pat, hits, rate, kept, keeprate in rows:
            w.writerow([pat, hits, f"{rate:.6f}", kept, f"{keeprate:.6f}", total])
    print(f"\nWrote: {path}")


def main():
    ap = argparse.ArgumentParser(description="Analyze Dollar Detective scan-data zips.")
    ap.add_argument("inputs", nargs="+", help="zip files and/or extracted folders")
    ap.add_argument("--out", metavar="DIR", help="also write pattern_hit_keep_rates.csv here")
    args = ap.parse_args()

    r = analyze(args.inputs)
    rows = print_report(r)
    if args.out and rows:
        write_csv(rows, r["total_rows"], args.out)


if __name__ == "__main__":
    main()
