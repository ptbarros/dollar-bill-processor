"""
Backup & restore for Dollar Detective user data.

Packages the irreplaceable, machine-local state into one portable .zip that can
be moved between Windows / macOS / Linux, and restores it selectively.

What's included (each an independently selectable "category"):
    settings     user_settings.yaml  (prefs, label profiles, pattern-label
                 overrides, rule tunings, custom patterns, colors, API keys)
    crop_config  config.yaml         (crop profiles / denomination)
    corrections  corrections.yaml    (hand-entered OCR fixes)
    patterns     patterns/user/**    (.lua scripts + sibling DataFile CSVs)
    ledger       ledger.db           (per-bill history — snapshotted, WAL folded in)

Deliberately excluded (regenerable or the user's own bulk files): session
recovery, the generated Insights HTML, debug logs, the review/ image folder, and
the archive folder of original scans. Those are documented for the user but not
swallowed into an app-state backup.

Cross-platform notes:
    * Everything is stored under a relative "data/" prefix + a manifest.json, so
      the archive is portable across OSes.
    * On restore, absolute paths inside user_settings.yaml that don't exist on
      the target machine are scrubbed, so a Windows backup restored on a Mac
      doesn't carry over dead C:\\ paths.
    * The ledger is snapshotted with SQLite's online-backup API rather than a raw
      file copy, so the -wal/-shm sidecars never matter.

Stdlib + PyYAML (already a dependency). Standalone: does not import the GUI.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sqlite3
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import yaml

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
CATEGORIES = ["settings", "crop_config", "corrections", "patterns", "ledger"]

CATEGORY_LABELS = {
    "settings": "Settings & preferences",
    "crop_config": "Crop profiles",
    "corrections": "OCR corrections",
    "patterns": "User patterns",
    "ledger": "Bill ledger (history)",
}

_SINGLE_FILES = {
    "settings": "user_settings.yaml",
    "crop_config": "config.yaml",
    "corrections": "corrections.yaml",
    "ledger": "ledger.db",
}


def _app_version() -> str:
    try:
        from version import __version__
        return __version__
    except Exception:
        return "unknown"


def _data_dir(data_dir=None) -> Path:
    if data_dir:
        return Path(data_dir)
    from resource_path import user_data_dir
    return user_data_dir()


def _patterns_dir(data_dir=None) -> Path:
    """User-pattern dir, resolved the same way pattern_engine_v3 does:
    under user_data_dir() when frozen, else patterns/user/ in the source tree."""
    if getattr(sys, "frozen", False):
        return _data_dir(data_dir) / "patterns" / "user"
    return Path(__file__).resolve().parent / "patterns" / "user"


def _snapshot_sqlite(src: Path, dst: Path) -> None:
    """Consistent copy of an open/live SQLite DB via the online-backup API."""
    s = sqlite3.connect(str(src))
    t = sqlite3.connect(str(dst))
    try:
        s.backup(t)
    finally:
        t.close()
        s.close()


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------
def create_backup(dest_zip, categories: Optional[Iterable[str]] = None,
                  include_api_keys: bool = True, data_dir=None, patterns_dir=None) -> dict:
    """Write a backup .zip. Returns the manifest dict (also stored in the zip).

    patterns_dir: the engine's real user-pattern directory; pass it from the GUI
    so dev/frozen resolution can't diverge. Falls back to the computed default.
    """
    dd = _data_dir(data_dir)
    pdir = Path(patterns_dir) if patterns_dir else _patterns_dir(data_dir)
    cats = list(categories) if categories is not None else list(CATEGORIES)
    manifest = {
        "schema": SCHEMA_VERSION,
        "app_version": _app_version(),
        "source_os": platform.system(),
        "created": datetime.now().isoformat(timespec="seconds"),
        "categories": {},
    }
    tmp = tempfile.mkdtemp(prefix="dbp_backup_")
    dest_zip = Path(dest_zip)
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for cat in cats:
                files = []
                if cat == "settings":
                    src = dd / "user_settings.yaml"
                    if src.exists():
                        if include_api_keys:
                            zf.write(src, "data/user_settings.yaml")
                        else:
                            data = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
                            ai = data.get("ai")
                            if isinstance(ai, dict):
                                for k in list(ai):
                                    if "key" in k.lower():
                                        ai[k] = ""
                            zf.writestr("data/user_settings.yaml",
                                        yaml.safe_dump(data, sort_keys=False))
                        files.append("user_settings.yaml")
                elif cat in ("crop_config", "corrections"):
                    src = dd / _SINGLE_FILES[cat]
                    if src.exists():
                        zf.write(src, f"data/{_SINGLE_FILES[cat]}")
                        files.append(_SINGLE_FILES[cat])
                elif cat == "ledger":
                    src = dd / "ledger.db"
                    if src.exists():
                        snap = Path(tmp) / "ledger.db"
                        _snapshot_sqlite(src, snap)
                        zf.write(snap, "data/ledger.db")
                        files.append("ledger.db")
                elif cat == "patterns":
                    if pdir.exists():
                        for f in sorted(pdir.rglob("*")):
                            if f.is_file() and "__pycache__" not in f.parts:
                                rel = f.relative_to(pdir).as_posix()
                                zf.write(f, f"data/patterns/user/{rel}")
                                files.append(f"patterns/user/{rel}")
                if files:
                    manifest["categories"][cat] = {"count": len(files), "files": files}
            zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return manifest


# ---------------------------------------------------------------------------
# Inspect
# ---------------------------------------------------------------------------
def read_manifest(zip_path) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read(MANIFEST_NAME))


def _looks_like_path(s: str) -> bool:
    """Absolute POSIX path, or a Windows drive path (C:\\...) seen on any OS."""
    if not s or len(s) < 2:
        return False
    if os.path.isabs(s):
        return True
    return len(s) >= 3 and s[1] == ":" and (s[2] in "\\/")


def _scrub_dead_paths(obj) -> int:
    """Blank absolute-path strings that don't exist on this machine (in place)."""
    n = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and _looks_like_path(v) and not Path(v).exists():
                obj[k] = ""; n += 1
            else:
                n += _scrub_dead_paths(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str) and _looks_like_path(v) and not Path(v).exists():
                obj[i] = ""; n += 1
            else:
                n += _scrub_dead_paths(v)
    return n


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------
def restore_backup(zip_path, categories: Optional[Iterable[str]] = None, *,
                   ledger_mode: str = "merge", data_dir=None, patterns_dir=None,
                   live_ledger=None) -> dict:
    """Restore selected categories from a backup .zip.

    settings/crop_config/corrections/patterns are replaced (patterns overwrite
    same-named files); settings also gets dead-path scrubbing. The ledger is
    'merge' (combine histories, default) or 'replace'. Returns a per-category
    report. `live_ledger` is the app's open Ledger, if any, so a merge/replace
    can use/close it safely.
    """
    dd = _data_dir(data_dir)
    pdir = Path(patterns_dir) if patterns_dir else _patterns_dir(data_dir)
    man = read_manifest(zip_path)
    present = set(man.get("categories", {}))
    cats = [c for c in (categories if categories is not None else present) if c in present]
    report = {}
    tmp = tempfile.mkdtemp(prefix="dbp_restore_")
    try:
        dd.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            for cat in cats:
                if cat == "settings":
                    data = yaml.safe_load(zf.read("data/user_settings.yaml")) or {}
                    scrubbed = _scrub_dead_paths(data)
                    (dd / "user_settings.yaml").write_text(
                        yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
                    report[cat] = f"replaced ({scrubbed} dead path(s) cleared)"
                elif cat in ("crop_config", "corrections"):
                    (dd / _SINGLE_FILES[cat]).write_bytes(
                        zf.read(f"data/{_SINGLE_FILES[cat]}"))
                    report[cat] = "replaced"
                elif cat == "patterns":
                    pdir.mkdir(parents=True, exist_ok=True)
                    n = 0
                    for name in zf.namelist():
                        if name.startswith("data/patterns/user/") and not name.endswith("/"):
                            rel = name[len("data/patterns/user/"):]
                            out = pdir / rel
                            out.parent.mkdir(parents=True, exist_ok=True)
                            out.write_bytes(zf.read(name))
                            n += 1
                    report[cat] = f"{n} file(s) restored"
                elif cat == "ledger":
                    snap = Path(tmp) / "ledger.db"
                    snap.write_bytes(zf.read("data/ledger.db"))
                    target = dd / "ledger.db"
                    if ledger_mode == "merge" and target.exists():
                        if live_ledger is not None:
                            res = live_ledger.merge_from(snap)
                        else:
                            from ledger import Ledger
                            lg = Ledger(target)
                            res = lg.merge_from(snap)
                            lg.close()
                        report[cat] = (f"merged (+{res['added_bills']} new bills, "
                                       f"{res['updated_bills']} updated)")
                    else:
                        if live_ledger is not None:
                            try:
                                live_ledger.close()
                            except Exception:
                                pass
                        shutil.copyfile(snap, target)
                        for sc in ("-wal", "-shm"):
                            p = Path(str(target) + sc)
                            if p.exists():
                                p.unlink()
                        report[cat] = "replaced"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return report


# ---------------------------------------------------------------------------
# CLI (handy for testing / headless use)
# ---------------------------------------------------------------------------
def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Dollar Detective backup/restore.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backup"); b.add_argument("dest"); b.add_argument("--data-dir")
    b.add_argument("--no-keys", action="store_true")
    r = sub.add_parser("restore"); r.add_argument("src"); r.add_argument("--data-dir")
    r.add_argument("--ledger", choices=["merge", "replace"], default="merge")
    sub.add_parser("info").add_argument("src")
    a = ap.parse_args()
    if a.cmd == "backup":
        m = create_backup(a.dest, include_api_keys=not a.no_keys, data_dir=a.data_dir)
        print(json.dumps(m, indent=2))
    elif a.cmd == "restore":
        print(json.dumps(restore_backup(a.src, ledger_mode=a.ledger, data_dir=a.data_dir), indent=2))
    elif a.cmd == "info":
        print(json.dumps(read_manifest(a.src), indent=2))


if __name__ == "__main__":
    _main()
