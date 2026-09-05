"""Pattern bundle export/import.

Pack one or more patterns -- together with any external data files they declare
via a ``DataFile:`` header (e.g. the ZIP-code or low-run CSV) -- into a single
shareable ``.ddpat`` file (a zip), and unpack them into the writable user
patterns directory so they can be shared between collectors.

Bundle layout (zip):
    manifest.json
    patterns/<file>.lua
    data/<datafile basename>      (only for patterns that declare a DataFile)

The manifest records, per pattern, the .lua arcname and the data-file arcname (if
any). On import each .lua goes into the user patterns dir and its data file is
written alongside it (a sibling); the pattern's ``DataFile:`` header is normalized
to the bare filename so it resolves next to the .lua regardless of how it was
declared in the source install.
"""

from __future__ import annotations

import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    from version import __version__
except Exception:  # pragma: no cover - version is always present in the app
    __version__ = "?"

BUNDLE_FORMAT = "dollar-detective-pattern-bundle"
BUNDLE_VERSION = 1
BUNDLE_EXT = ".ddpat"


def _resolve_data_file(engine, info) -> Path | None:
    """Absolute path to a pattern's DataFile, or None if it has none / is missing.

    Mirrors the engine's own resolution: a ``data/...`` path looks under
    ``patterns/data/``; anything else is a sibling of the .lua file."""
    df = (getattr(info, "data_file", "") or "").strip()
    if not df:
        return None
    if df.startswith("data/"):
        p = engine.patterns_dir / df
    else:
        p = Path(info.file_path).parent / df
    return p if p.exists() else None


def export_bundle(engine, pattern_names, out_path) -> dict:
    """Write the named patterns (and their data files) to a ``.ddpat`` bundle.

    Returns a summary dict: count, data_files, missing_data (list of
    (pattern, declared_datafile) whose file couldn't be found), path."""
    out_path = Path(out_path)
    entries = []
    missing_data = []
    used_lua_names: set[str] = set()
    used_data_names: set[str] = set()

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in pattern_names:
            info = engine.lua_patterns.get(name)
            if info is None:
                continue
            lua_path = Path(info.file_path)
            if not lua_path.exists():
                continue

            # Unique arcname for the .lua (avoid clobbering same-named files).
            arc = lua_path.name
            stem, ext = lua_path.stem, lua_path.suffix
            i = 1
            while arc in used_lua_names:
                arc = f"{stem}_{i}{ext}"
                i += 1
            used_lua_names.add(arc)

            data_arc = None
            data_name = None
            declared = (getattr(info, "data_file", "") or "").strip()
            df_abs = _resolve_data_file(engine, info)
            if declared and df_abs is None:
                missing_data.append((name, declared))
            if df_abs is not None:
                data_name = df_abs.name
                if data_name not in used_data_names:
                    z.write(df_abs, f"data/{data_name}")
                    used_data_names.add(data_name)
                data_arc = f"data/{data_name}"

            z.write(lua_path, f"patterns/{arc}")
            entries.append({
                "name": name,
                "display_name": getattr(info, "display_name", "") or "",
                "library": getattr(info, "library", "user"),
                "file": f"patterns/{arc}",
                "data_file": data_arc,
                "data_file_name": data_name,
            })

        manifest = {
            "format": BUNDLE_FORMAT,
            "version": BUNDLE_VERSION,
            "app_version": __version__,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "patterns": entries,
        }
        z.writestr("manifest.json", json.dumps(manifest, indent=2))

    return {
        "count": len(entries),
        "data_files": len(used_data_names),
        "missing_data": missing_data,
        "path": str(out_path),
    }


def read_manifest(bundle_path) -> dict:
    """Read + validate a bundle's manifest. Raises ValueError if it isn't one."""
    with zipfile.ZipFile(bundle_path, "r") as z:
        try:
            manifest = json.loads(z.read("manifest.json"))
        except KeyError:
            raise ValueError("Not a valid pattern bundle (no manifest.json).")
        except Exception as e:
            raise ValueError(f"Could not read the bundle: {e}")
    if not isinstance(manifest, dict) or manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError("That doesn't look like a Dollar Detective pattern bundle.")
    return manifest


def bundle_collisions(engine, manifest) -> list[str]:
    """Names in the manifest that already exist in the engine or on disk."""
    dest = Path(engine.user_patterns_dir)
    hits = []
    for entry in manifest.get("patterns", []):
        name = entry.get("name", "")
        lua_basename = Path(entry.get("file", "")).name
        if not lua_basename:
            continue
        if name in engine.lua_patterns or (dest / lua_basename).exists():
            hits.append(name)
    return hits


def import_bundle(engine, bundle_path, overwrite=False) -> dict:
    """Extract a bundle's patterns + data files into the user patterns dir.

    Existing patterns (by name or target filename) are skipped unless
    ``overwrite`` is True. Calls ``engine.reload()`` at the end. Returns a summary
    dict: imported, skipped, errors, data_files."""
    dest = Path(engine.user_patterns_dir)
    dest.mkdir(parents=True, exist_ok=True)

    imported, skipped, errors = [], [], []
    data_written: set[str] = set()

    manifest = read_manifest(bundle_path)

    with zipfile.ZipFile(bundle_path, "r") as z:
        names_in_zip = set(z.namelist())
        for entry in manifest.get("patterns", []):
            name = entry.get("name", "")
            lua_arc = entry.get("file", "")
            if not lua_arc or lua_arc not in names_in_zip:
                errors.append(f"{name or lua_arc}: missing from bundle")
                continue
            lua_basename = Path(lua_arc).name  # basename only: no path traversal
            script = z.read(lua_arc).decode("utf-8", errors="replace")

            # Data file (if any): write it as a sibling and normalize the header.
            data_arc = entry.get("data_file")
            data_basename = None
            if data_arc and data_arc in names_in_zip:
                data_basename = Path(data_arc).name
                script = re.sub(
                    r'(?im)^(\s*DataFile:\s*).*$',
                    lambda mm: mm.group(1) + data_basename, script)
            elif data_arc:
                errors.append(f"{name}: data file missing from bundle")

            target = dest / lua_basename
            exists = target.exists() or (name in engine.lua_patterns)
            if exists and not overwrite:
                skipped.append(name)
                continue

            try:
                if data_basename is not None:
                    (dest / data_basename).write_bytes(z.read(data_arc))
                    data_written.add(data_basename)
                target.write_text(script, encoding="utf-8")
                imported.append(name)
            except Exception as e:
                errors.append(f"{name}: {e}")

    engine.reload()
    return {
        "imported": imported,
        "skipped": skipped,
        "errors": errors,
        "data_files": len(data_written),
    }
