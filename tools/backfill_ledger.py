#!/usr/bin/env python3
"""
Backfill the bill ledger from historical scan data.

Seeds ledger.db from collected `DBP_data_*.zip` files (from tools/collect_scans.bat)
and/or already-extracted month folders, so the app's Insights report and
"seen before?" history include everything processed before the ledger went live.

Usage:
    python tools/backfill_ledger.py DBP_data_August_*.zip DBP_data_September_*.zip
    python tools/backfill_ledger.py "/path/to/August Scans"      # a folder of straps
    python tools/backfill_ledger.py *.zip --db /path/to/ledger.db
    python tools/backfill_ledger.py *.zip --dry-run              # report, don't write

Notes:
  * One session is created per strap folder; re-runs are collapsed to the newest
    results_*.csv per folder (same dedup as tools/analyze_scans.py).
  * "kept" is reconstructed from crop filenames (<SERIAL>_NN.jpg) in each strap's
    _all_files_manifest.csv (or loose crop files if present).
  * Identity is serial-only (series_year left blank) so historical crops match
    their bills regardless of whether plate extraction was on. Live scanning
    records the real series_year.
  * Safe to re-run: bumps times_seen on bills already present rather than
    duplicating them. Prefer a fresh DB (or --db to a scratch path) the first time.

Dependency-free apart from ledger.py (stdlib only).
"""

import argparse
import csv
import io
import re
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ledger import Ledger, normalize_serial  # noqa: E402

CROP_RE = re.compile(r"^(?P<serial>.+?)_\d{2,}\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def _default_db() -> Path:
    try:
        from resource_path import user_data_dir
        return user_data_dir() / "ledger.db"
    except Exception:
        return Path.cwd() / "ledger.db"


def _extract_zip(zp: Path, tmp: str) -> Path:
    """Extract a (possibly Windows-made, backslash-path) zip to a temp dir."""
    dest = Path(tmp) / zp.stem
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            out = dest / info.filename.replace("\\", "/")
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(out, "wb") as f:
                f.write(src.read())
    return dest


def _sources(inputs, tmp):
    for item in inputs:
        p = Path(item)
        if not p.exists():
            print(f"  ! skipping (not found): {p}", file=sys.stderr)
            continue
        if p.is_dir():
            yield p.name, p
        elif p.suffix.lower() == ".zip":
            try:
                yield p.stem, _extract_zip(p, tmp)
            except zipfile.BadZipFile:
                print(f"  ! skipping (bad zip): {p}", file=sys.stderr)
        else:
            print(f"  ! skipping (not folder/zip): {p}", file=sys.stderr)


def _latest_per_folder(root: Path):
    latest = {}
    for csv_path in root.rglob("results_*.csv"):
        key = str(csv_path.parent)
        if key not in latest or csv_path.name > latest[key].name:
            latest[key] = csv_path
    return latest


def _kept_serials(root: Path):
    kept = set()
    for man in root.rglob("_all_files_manifest.csv"):
        with open(man, newline="", encoding="utf-8-sig", errors="replace") as f:
            for row in csv.DictReader(f):
                m = CROP_RE.match((row.get("Name") or "").strip())
                if m:
                    kept.add(normalize_serial(m.group("serial")))
    for img in root.rglob("*.jpg"):
        m = CROP_RE.match(img.name)
        if m:
            kept.add(normalize_serial(m.group("serial")))
    return kept


def backfill(inputs, db_path, dry_run=False):
    tmp = tempfile.mkdtemp(prefix="dbp_backfill_")
    led = Ledger(db_path)
    recorded = 0
    kept_all = set()

    for label, root in _sources(inputs, tmp):
        latest = _latest_per_folder(root)
        print(f"  {label}: {len(latest)} strap folders")
        for csv_path in sorted(latest.values()):
            folder = csv_path.parent.name
            sid = led.start_session(source_folder=folder, label=f"{label}/{folder}",
                                    app_version="backfill")
            with open(csv_path, newline="", encoding="utf-8-sig", errors="replace") as f:
                for row in csv.DictReader(f):
                    if not (row.get("serial") or "").strip():
                        continue
                    led.record(
                        sid,
                        serial=row.get("serial"),
                        series_year="",   # serial-only identity for history
                        patterns=row.get("fancy_types"),
                        confidence=_f(row.get("confidence")),
                        needs_review=(row.get("needs_review", "").strip().lower()
                                      in ("true", "1", "yes")),
                        error=(row.get("error") or None),
                        front_file=row.get("front_file"),
                        back_file=row.get("back_file"),
                        front_plate=row.get("front_plate", "") or "",
                        back_plate=row.get("back_plate", "") or "",
                        potential_mule=(row.get("potential_mule", "").strip().lower()
                                        in ("true", "1", "yes")),
                    )
                    recorded += 1
        kept_all |= _kept_serials(root)

    n_kept = led.mark_kept(kept_all) if not dry_run else 0
    st = led.stats()
    led.close()

    print("\n" + "=" * 52)
    print(f"  Backfill {'(DRY RUN — kept not written)' if dry_run else 'complete'}")
    print("=" * 52)
    print(f"  Rows recorded      : {recorded:,}")
    print(f"  Unique bills       : {st['unique_bills']:,}")
    print(f"  Observations       : {st['observations']:,}")
    print(f"  Fancy              : {st['fancy']:,}")
    print(f"  Crop serials found : {len(kept_all):,}")
    if not dry_run:
        print(f"  Bills marked kept  : {st['kept']:,}")
    print(f"  DB: {db_path}")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description="Backfill ledger.db from scan zips/folders.")
    ap.add_argument("inputs", nargs="+", help="DBP_data_*.zip files and/or month folders")
    ap.add_argument("--db", default=str(_default_db()), help="ledger.db path")
    ap.add_argument("--dry-run", action="store_true", help="record bills but don't mark kept / report only")
    args = ap.parse_args()
    print(f"Backfilling ledger: {args.db}")
    backfill(args.inputs, args.db, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
