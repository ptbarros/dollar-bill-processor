#!/usr/bin/env python3
"""Assign the next catalog SKU (C-###) to core patterns that don't have one.

Catalog numbers are STABLE, append-only identifiers (see the catalog scheme in
CLAUDE.md): existing C-### are never renumbered, and a newly-added core pattern
takes the next free number -- NOT a slot matching its rarity. Rarity ordering is
a separate derived view. This is the maintainer tool for adding a core pattern
(the in-app create flow numbers USER patterns U-### automatically via
engine.next_catalog_sku).

Dry run by default; pass --write to edit the headers.

    python tools/assign_catalog.py            # show what would be assigned
    python tools/assign_catalog.py --write     # bake the new C-### headers
"""
import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE = PROJECT_ROOT / "patterns" / "core"

CATALOG_RE = re.compile(r"^\s*Catalog:\s*(.+)$", re.MULTILINE)
PATTERN_RE = re.compile(r"^(\s*)Pattern:\s*(.+)$", re.MULTILINE)
DISPLAY_RE = re.compile(r"^(\s*)DisplayName:\s*.*$", re.MULTILINE)
HEADER_RE = re.compile(r"--\[\[(.*?)--\]\]", re.DOTALL)
SKU_RE = re.compile(r"^C-(\d+)$")


def insert_catalog(text: str, sku: str) -> str:
    """Insert 'Catalog: <sku>' after DisplayName (or Pattern) in the header."""
    m = DISPLAY_RE.search(text) or PATTERN_RE.search(text)
    indent = m.group(1)
    return text[:m.end()] + f"\n{indent}Catalog: {sku}" + text[m.end():]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="edit the .lua headers (default: dry run)")
    args = ap.parse_args()

    files = sorted(CORE.glob("*.lua"))
    max_n = 0
    unnumbered = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        if not HEADER_RE.search(text):
            print(f"  SKIP (no header): {f.name}")
            continue
        m = CATALOG_RE.search(text)
        sku_m = SKU_RE.match(m.group(1).strip()) if m else None
        if sku_m:
            max_n = max(max_n, int(sku_m.group(1)))
        elif not m:
            unnumbered.append(f)
        # A non-C- catalog value is left alone (shouldn't happen for core).

    if not unnumbered:
        print(f"All {len(files)} core patterns already have a C-### SKU "
              f"(highest C-{max_n:03d}). Nothing to assign.")
        return

    print(f"Highest existing SKU: C-{max_n:03d}. "
          f"Assigning {len(unnumbered)} new SKU(s):")
    n = max_n
    for f in unnumbered:
        n += 1
        sku = f"C-{n:03d}"
        print(f"  {sku}  <-  {f.name}")
        if args.write:
            f.write_text(insert_catalog(f.read_text(encoding="utf-8"), sku),
                         encoding="utf-8")
    if args.write:
        print(f"\nWrote {len(unnumbered)} header(s).")
    else:
        print("\n(dry run -- pass --write to bake these into the headers)")


if __name__ == "__main__":
    main()
