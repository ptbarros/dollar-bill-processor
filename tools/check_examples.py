#!/usr/bin/env python3
"""
check_examples.py -- assert every shipped pattern has a self-matching header example.

For each name in patterns/shipped_enabled.json, at least one of the pattern's
`Examples:` header serials must actually match the pattern when run through the
engine. This catches two traps a header can hide (both real, found 2026-09-21):
  * examples in the wrong form -- e.g. DUPLICATE_SN needs the "B...*" star wrapper,
    so bare-digit examples render an empty preview;
  * no `Examples:` line at all -- e.g. PYRAMID_LADDER once kept them only in prose.

Some patterns legitimately have no digit example (they key on the physical note or
a data file), listed in EXEMPT.

Exit 0 if all good, 1 if any pattern fails. Engine-only, no ODS needed:
    ./venv/bin/python tools/check_examples.py
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Patterns that legitimately match on something other than the 8 digits (physical
# note / data file), so a self-matching digit example isn't expected.
EXEMPT = {"GAS_PUMP", "SEAL_SHIFT", "LOW_RUN_6M", "LOW_RUN_12M"}

# BookRef is load-bearing: it's the sole marker of Green Guide origin after the
# single-core flatten, and the web app keys its book credit off it. Exactly the
# 45 book patterns should carry one; if this count changes, a rename/edit dropped
# (or spuriously added) a BookRef -- update this only on a deliberate set change.
EXPECTED_BOOKREFS = 45


def main() -> int:
    from pattern_engine_v3 import PatternEngineV3

    eng = PatternEngineV3()
    shipped = json.loads((PROJECT_ROOT / "patterns" / "shipped_enabled.json").read_text())

    no_examples, no_match, missing = [], [], []
    for name in shipped:
        info = eng.lua_patterns.get(name)
        if info is None:
            missing.append(name)
            continue
        examples = info.examples or []
        if not examples:
            if name not in EXEMPT:
                no_examples.append(name)
            continue
        if not any(name in eng.classify_simple(s) for s in examples):
            no_match.append((name, examples))

    book = [n for n in shipped if getattr(eng.lua_patterns.get(n), "book_ref", "")]
    bookref_bad = len(book) != EXPECTED_BOOKREFS

    ok = not (no_examples or no_match or missing or bookref_bad)
    checked = len(shipped) - len(EXEMPT)
    if ok:
        print(f"OK: all {checked} non-exempt shipped patterns have a self-matching example "
              f"({len(EXEMPT)} exempt); {len(book)} carry a BookRef.")
        return 0

    if missing:
        print("!! shipped names not loaded by the engine:")
        for n in missing:
            print(f"     {n}")
    if no_examples:
        print("!! shipped patterns with NO Examples: header line:")
        for n in no_examples:
            print(f"     {n}")
    if no_match:
        print("!! shipped patterns whose examples never match the pattern:")
        for n, ex in no_match:
            print(f"     {n}: {ex}")
    if bookref_bad:
        print(f"!! BookRef count is {len(book)}, expected {EXPECTED_BOOKREFS} "
              f"(a rename/edit changed which patterns carry the licensing credit).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
