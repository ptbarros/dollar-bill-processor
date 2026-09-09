#!/usr/bin/env python3
"""
Build a standalone `patterns/Essentials/` LIBRARY from an Essentials selection.

FIL thinks in libraries (folders in Pattern Manager), not in enable/disable
selections. This copies every pattern enabled in a selection JSON into a new
`patterns/Essentials/` folder so it shows up as one "Essentials" library he can
enable-all and A/B against his normal set — no presets to explain.

Copies keep the FAMILIAR display name (so FIL sees "Solid", "Radar", ...) but
get a UNIQUE internal name (ESS_ prefix) so they never collide with — and
silently overwrite — the originals in core/Nicks/Green Guide/user. Data files
are copied alongside and the DataFile header normalized to the basename.

This is a duplicate (test) artifact: only ONE of {Essentials, the originals}
should be enabled at a time (else every bill double-fires). For FIL's A/B that
means enable the Essentials library / disable the rest, then the reverse.

Usage:
    python3 tools/build_essentials_library.py                 # uses ~/Documents/Essentials_proposed.json
    python3 tools/build_essentials_library.py path/to/selection.json
    python3 tools/build_essentials_library.py --clean         # wipe the folder first

Dependency-free apart from pattern_engine_v3 (stdlib only).
"""

import argparse
import csv
import difflib
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pattern_engine_v3 import PatternEngineV3  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEST = REPO / "patterns" / "Essentials"
DEFAULT_SELECTION = Path("~/Documents/Essentials_proposed.json").expanduser()
PREFIX = "ESS_"

_PATTERN_RE = re.compile(r"(?im)^(\s*Pattern:\s*)(\S+)\s*$")
_DISPLAY_RE = re.compile(r"(?im)^\s*DisplayName:\s*(.+?)\s*$")
_DATAFILE_RE = re.compile(r"(?im)^(\s*DataFile:\s*)(.+?)\s*$")
_DESC_RE = re.compile(r"(?im)^(\s*Description:\s*)(.+?)\s*$")


def _concept(text):
    """A pattern's bare concept name for matching (drop CS- prefix, normalize)."""
    return re.sub(r"^cs[- ]", "", (text or "").strip().lower()).replace("_", " ").strip()


def build_gg_index(eng):
    """{concept: (book_ref, display)} over Green Guide patterns, + concept list."""
    idx = {}
    for n, i in eng.lua_patterns.items():
        if getattr(i, "library", "") == "The Green Guide":
            c = _concept(i.display_name or n)
            if c and c not in idx:
                idx[c] = (getattr(i, "book_ref", "") or "", i.display_name or n)
    return idx, list(idx)


def gg_reference(info, gg_index, gg_concepts):
    """The Green Guide reference line for a pattern, or (None, None).
    GG patterns cite themselves; others fuzzy-match to the closest GG concept.
    Returns (text, kind) where kind is 'exact' or 'approx'."""
    if getattr(info, "book_ref", "") and getattr(info, "library", "") == "The Green Guide":
        return f"Green Guide: {info.book_ref} ({info.display_name or info.name})", "exact"
    c = _concept(info.display_name or info.name)
    # Conservative: only a near-identical concept name earns an (approx) ref —
    # a wrong Green Guide reference is worse than none (e.g. Star Note must NOT
    # match "Year Notes"). Exact concept equality always wins.
    if c in gg_index:
        ref, disp = gg_index[c]
        return f"~ Green Guide: {ref} ({disp})".strip(), "approx"
    hit = difflib.get_close_matches(c, gg_concepts, n=1, cutoff=0.86)
    if hit:
        ref, disp = gg_index[hit[0]]
        return f"~ Green Guide: {ref} ({disp})".strip(), "approx"
    return None, None


def find_data_file(basename, src_dir):
    """Locate a pattern's data file by basename near its source."""
    for cand in (src_dir / basename,
                 REPO / "patterns" / "data" / basename,
                 REPO / "patterns" / basename):
        if cand.exists():
            return cand
    # last resort: search the whole patterns tree
    hits = list((REPO / "patterns").rglob(basename))
    return hits[0] if hits else None


def transform(text, info, gg_ref=None):
    """Rewrite a pattern file: unique internal name, ensured familiar
    DisplayName, basename DataFile, and a Green Guide reference appended to the
    Description. Returns (new_text, data_basename|None)."""
    friendly = info.display_name or info.name

    # Unique internal name.
    text, n = _PATTERN_RE.subn(
        lambda m: f"{m.group(1)}{PREFIX}{m.group(2)}", text, count=1)
    if not n:  # no header Pattern line — shouldn't happen, but be safe
        text = f"--[[\nPattern: {PREFIX}{info.name}\n--]]\n" + text

    # Ensure a DisplayName so FIL sees the friendly name, not ESS_*.
    if not _DISPLAY_RE.search(text):
        text = re.sub(r"(?im)^(\s*Pattern:\s*\S+[ \t]*)$",
                      lambda m: f"{m.group(1)}\nDisplayName: {friendly}",
                      text, count=1)

    # Append the Green Guide reference to the Description so it shows in the app.
    if gg_ref:
        if _DESC_RE.search(text):
            text = _DESC_RE.sub(
                lambda m: f"{m.group(1)}{m.group(2)}  |  {gg_ref}", text, count=1)
        else:  # no Description line — add one after DisplayName/Pattern
            text = re.sub(r"(?im)^(\s*DisplayName:\s*.+?[ \t]*)$",
                          lambda m: f"{m.group(1)}\nDescription: {gg_ref}",
                          text, count=1)

    # Normalize DataFile to its basename (we copy it as a sibling).
    data_basename = None
    mdf = _DATAFILE_RE.search(text)
    if mdf:
        data_basename = Path(mdf.group(2)).name
        text = _DATAFILE_RE.sub(
            lambda m: f"{m.group(1)}{data_basename}", text, count=1)
    return text, data_basename


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("selection", nargs="?", default=str(DEFAULT_SELECTION),
                    help=f"Selection JSON. Default {DEFAULT_SELECTION}")
    ap.add_argument("--clean", action="store_true",
                    help="Remove the Essentials folder before building.")
    args = ap.parse_args(argv)

    sel_path = Path(args.selection).expanduser()
    if not sel_path.exists():
        ap.error(f"Selection not found: {sel_path}")
    states = json.loads(sel_path.read_text()).get("pattern_states", {})
    enabled = {n for n, on in states.items() if on}
    print(f"Selection: {sel_path.name} — {len(enabled)} enabled patterns")

    eng = PatternEngineV3()
    gg_index, gg_concepts = build_gg_index(eng)

    if args.clean and DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True, exist_ok=True)

    written, data_copied, missing = 0, 0, []
    ref_rows, ref_counts = [], {"exact": 0, "approx": 0, "none": 0}
    for name in sorted(enabled):
        info = eng.lua_patterns.get(name)
        if info is None or not getattr(info, "file_path", None):
            missing.append(name)
            continue
        src = Path(info.file_path)
        if not src.exists():
            missing.append(name)
            continue
        gg_ref, kind = gg_reference(info, gg_index, gg_concepts)
        ref_counts[kind or "none"] += 1
        ref_rows.append((name, info.display_name or name,
                         getattr(info, "library", ""), kind or "none",
                         gg_ref or ""))
        new_text, data_basename = transform(src.read_text(encoding="utf-8"),
                                            info, gg_ref)
        (DEST / f"{PREFIX}{name}.lua").write_text(new_text, encoding="utf-8")
        written += 1
        if data_basename:
            data_src = find_data_file(data_basename, src.parent)
            if data_src and data_src.exists():
                shutil.copy2(data_src, DEST / data_basename)
                data_copied += 1
            else:
                print(f"  ! data file not found for {name}: {data_basename}")

    # Green Guide reference report for review/correction.
    report = DEST.parent.parent / "essentials_gg_references.csv"
    with report.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pattern", "display", "source_library", "gg_match_kind",
                    "green_guide_reference"])
        w.writerows(sorted(ref_rows, key=lambda r: (r[3], r[1])))

    print(f"Wrote {written} patterns + {data_copied} data files -> {DEST}/")
    print(f"Green Guide refs: {ref_counts['exact']} exact (GG patterns), "
          f"{ref_counts['approx']} approx (fuzzy match), "
          f"{ref_counts['none']} none -> {report}")
    if missing:
        print(f"  ! {len(missing)} not found in engine (skipped): "
              + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
