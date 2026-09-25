"""Review bundle (.zip) export / import.

The "Save for Review" action already accumulates flagged bills in one review
folder across many straps: per-bill front/back/serial-crop images (timestamp
-prefixed so they never collide) plus a growing ``review_log.csv`` that records
what the model detected (serial, confidence, patterns) and the reviewer's note.

This module packs that folder into a single portable ``.zip`` the FIL emails to
Paul (one-way), and reads it back for an in-app viewer. It intentionally mirrors
the offline-portable style of ``pattern_bundle.py`` / ``backup_manager.py`` --
no accounts, no server.
"""

import csv
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

MANIFEST_NAME = "review_manifest.json"
LOG_NAME = "review_log.csv"
SCHEMA = "dollar-detective-review-bundle/1"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def review_bundle_stats(review_folder) -> dict:
    """Count reviewable content in `review_folder` (drives the export prompt)."""
    folder = Path(review_folder)
    if not folder.is_dir():
        return {"exists": False, "rows": 0, "images": 0}
    rows = 0
    log = folder / LOG_NAME
    if log.exists():
        try:
            with open(log, newline="", encoding="utf-8") as f:
                rows = max(0, sum(1 for _ in csv.reader(f)) - 1)  # minus header
        except Exception:
            rows = 0
    images = sum(1 for p in folder.iterdir()
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return {"exists": True, "rows": rows, "images": images}


def export_review_bundle(review_folder, dest_zip, app_version: str = "") -> dict:
    """Zip the review folder's files (images + review_log.csv) plus a manifest.

    Only top-level files are packed (an already-archived ``sent/`` subfolder is
    left out). Returns the manifest dict.
    """
    folder = Path(review_folder)
    dest_zip = Path(dest_zip)
    dest_zip.parent.mkdir(parents=True, exist_ok=True)

    files = []
    if folder.is_dir():
        files = [p for p in sorted(folder.iterdir())
                 if p.is_file() and p.name != MANIFEST_NAME]

    stats = review_bundle_stats(folder)
    manifest = {
        "schema": SCHEMA,
        "created": datetime.now().isoformat(timespec="seconds"),
        "app_version": app_version,
        "rows": stats["rows"],
        "images": stats["images"],
        "files": [p.name for p in files],
    }
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, p.name)
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
    return manifest


def archive_review_folder(review_folder, mode: str = "sent") -> int:
    """After a successful export, tidy the folder so the next batch starts fresh.

    ``mode="sent"`` moves the current top-level files into a timestamped
    ``sent/<ts>/`` subfolder (nothing is destroyed); ``mode="delete"`` removes
    them. Returns the number of files affected.
    """
    folder = Path(review_folder)
    if not folder.is_dir():
        return 0
    files = [p for p in folder.iterdir() if p.is_file()]
    if not files:
        return 0
    if mode == "delete":
        n = 0
        for p in files:
            try:
                p.unlink()
                n += 1
            except Exception:
                pass
        return n
    # "sent": move into a timestamped archive subfolder
    dest = folder / "sent" / datetime.now().strftime("%Y%m%d_%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in files:
        try:
            shutil.move(str(p), str(dest / p.name))
            n += 1
        except Exception:
            pass
    return n


def _classify_files(names: List[str], base: Path):
    """Split a row's copied files into (front, back, serial_crop). The save writes
    them in order [front, back, serial-crop], and the serial crop is tagged with
    '_serial_' in its name."""
    front = back = serial_crop = None
    plains = []
    for name in names:
        p = base / name
        if not p.exists():
            continue
        if "_serial_" in name:
            serial_crop = str(p)
        else:
            plains.append(str(p))
    if plains:
        front = plains[0]
    if len(plains) > 1:
        back = plains[1]
    return front, back, serial_crop


def read_review_bundle(zip_path, extract_dir) -> dict:
    """Extract a bundle and parse review_log.csv into viewer-ready items.

    Returns {'dir', 'manifest', 'items': [{timestamp, serial, note, confidence,
    patterns, front, back, serial_crop, files}...]}.
    """
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    manifest = {}
    mp = extract_dir / MANIFEST_NAME
    if mp.exists():
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    items = []
    log = extract_dir / LOG_NAME
    if log.exists():
        with open(log, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                names = [s.strip() for s in (row.get("files_copied") or "").split(";")
                         if s.strip()]
                front, back, serial_crop = _classify_files(names, extract_dir)
                items.append({
                    "timestamp": row.get("timestamp", ""),
                    "serial": row.get("serial", ""),
                    "note": row.get("note", ""),
                    "confidence": row.get("confidence", ""),
                    "patterns": row.get("patterns", ""),
                    "front_file": row.get("front_file", ""),
                    "front": front,
                    "back": back,
                    "serial_crop": serial_crop,
                    "files": [str(extract_dir / n) for n in names
                              if (extract_dir / n).exists()],
                })
    return {"dir": str(extract_dir), "manifest": manifest, "items": items}
