#!/usr/bin/env python3
"""
compute_rarity.py -- exact serial-number rarity for digit-deterministic patterns.

Approach (see memory web-features-port-assessment, "RARITY+TIER FEATURE"):
brute-force the printed serial range with the REAL Lua match() of each pattern,
compiled ONCE per worker (not recompiled per call the way pattern_sandbox does).
The Lua IS the ground truth, so there is no separate numpy reimplementation to
verify -- counts are exact by construction.

Rarity = exact count of matching serials in the printed range [MIN, MAX]
(default [1, 96_000_000]; the top ~4M serials of a 10^8 block are never printed),
reported as "1 in X" where X = range_size / count.

Non-digit-deterministic patterns are skipped (their odds stay hand-set):
  * DataFile-backed (ZIP_CODE, LOW_RUN_6M/12M, KNOWN_SERIALS, SPECIAL_DATES, ...)
  * image/metadata patterns (GAS_PUMP, SEAL_SHIFT)
  * date-relative patterns (reference ctx.metadata.current_year/month/day)

Run under `heavy` for the full sweep, e.g.:
    heavy ./venv/bin/python tools/compute_rarity.py --out ~/Documents/rarity.csv

MUST run with ./venv/bin/python (needs lupa; numpy optional).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

# Make the project importable regardless of CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MIN = 1
DEFAULT_MAX = 96_000_000  # top ~4M serials of a 10^8 block are never printed

# Patterns we never brute-force (odds stay hand-set); matched in addition to the
# auto-detected DataFile / date-relative ones.
# STAR matches on the star SUFFIX, not the 8 digits, so a digit-only brute force
# never triggers it (would report a false 0) -- skip like the other physical ones.
HARD_SKIP = {"GAS_PUMP", "SEAL_SHIFT", "DUPLICATE_SN", "STAR"}
# Lua tokens that mark a pattern as date-relative (its count shifts with "today").
DATE_TOKENS = ("current_year", "current_month", "current_day")

# Lua wrapper: define the pattern's match(), capture it in a LOCAL (so each
# pattern's closure calls ITS OWN match, not whatever global `match` was defined
# last), then return a closure yielding a plain boolean -- avoids per-call table
# attribute access from Python.
_WRAP = "\nlocal __m = match; return function(c) local r = __m(c); return (r and r.matched) and true or false end"


# --- worker globals (populated by _init) ----------------------------------
_W_NAMES: list[str] = []
_W_FNS: list = []
_W_BUILD_CTX = None


def _init(pattern_scripts):
    """Per-worker: build one LuaRuntime and compile every pattern's match once."""
    global _W_NAMES, _W_FNS, _W_BUILD_CTX
    import lupa

    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    # Load the helper library as globals (patterns call is_ladder/is_palindrome/
    # find_runs/... as globals, and log() as a no-op when not debugging).
    lua.execute("function log(...) end")
    helpers_path = PROJECT_ROOT / "patterns" / "lib" / "helpers.lua"
    if helpers_path.exists():
        lua.execute(helpers_path.read_text(encoding="utf-8"))
    # Build the ctx table entirely in Lua so no Python->Lua table conversion
    # happens per serial.
    _W_BUILD_CTX = lua.eval(
        "function(n)"
        "  local s = string.format('%08d', n)"
        "  local dl = {}"
        "  for i=1,8 do dl[i] = tonumber(string.sub(s,i,i)) end"
        "  return {digits=s, full_serial=s, digit_list=dl, metadata={}}"
        "end"
    )
    _W_NAMES = []
    _W_FNS = []
    for name, script in pattern_scripts:
        try:
            fn = lua.execute(script + _WRAP)
        except Exception as e:  # pragma: no cover - surfaced at load time in main
            raise RuntimeError(f"failed to compile pattern {name}: {e}")
        _W_NAMES.append(name)
        _W_FNS.append(fn)


def _count_chunk(rng):
    """Count matches per pattern over serials [start, end] inclusive."""
    start, end = rng
    fns = _W_FNS
    build = _W_BUILD_CTX
    counts = [0] * len(fns)
    for n in range(start, end + 1):
        ctx = build(n)
        for i, fn in enumerate(fns):
            if fn(ctx):
                counts[i] += 1
    return counts


# --- pattern selection -----------------------------------------------------
def load_targets(shipped_only=True, only=None, include_skipped=False):
    """Return (targets, skipped) where each is list of (name, script, info)."""
    from pattern_engine_v3 import PatternEngineV3

    eng = PatternEngineV3()
    installed = eng.lua_patterns

    if only:
        wanted = [n for n in only if n in installed]
        missing = [n for n in only if n not in installed]
        if missing:
            print(f"WARNING: --only names not installed: {missing}", file=sys.stderr)
        names = wanted
    elif shipped_only:
        shipped = json.loads((PROJECT_ROOT / "patterns" / "shipped_enabled.json").read_text())
        names = [n for n in shipped if n in installed]
        missing = [n for n in shipped if n not in installed]
        if missing:
            print(f"WARNING: shipped names not installed: {missing}", file=sys.stderr)
    else:
        names = list(installed.keys())

    targets, skipped = [], []
    for name in names:
        info = installed[name]
        reason = _skip_reason(name, info)
        if reason and not include_skipped:
            skipped.append((name, reason))
        else:
            targets.append((name, info.script))
    return targets, skipped, eng


def _skip_reason(name, info):
    if name in HARD_SKIP:
        return "hard-skip (image/metadata)"
    if getattr(info, "data", None) is not None:
        return "DataFile-backed"
    script = info.script or ""
    if any(tok in script for tok in DATE_TOKENS):
        return "date-relative"
    return None


# --- tier derivation -------------------------------------------------------
def derive_tier(one_in_x):
    """Log buckets ~x5 apart. Tier 1 = rarest (matches codebase convention:
    SOLID=1, RADAR=3, BINARY=4). Rarer (larger X) -> lower tier number.
    Clamped to [1, 10]. Returns None for a meaningless denominator (0 matches):
    a count-0 pattern has NO real tier, so never derive one (else it lands at
    tier 10 = most common, exactly backwards)."""
    if one_in_x is None:
        return None
    if one_in_x <= 1:
        return 10
    # log5(X): 1-in-5 -> ~1 bucket ... 1-in-~10M -> ~10 buckets.
    buckets = math.log(one_in_x, 5)
    tier = 11 - int(round(buckets))
    return max(1, min(10, tier))


# --- header rewrite --------------------------------------------------------
_ODDS_RE = re.compile(r"^(\s*Odds:\s*).*$", re.MULTILINE)


def odds_string(one_in_x: int, count: int) -> str:
    """Chosen display format (Paul 2026-09-20): familiar '1 in N' plus the exact
    raw count so there is zero hidden rounding. dd-web renders this string
    verbatim, so the count must live in the string itself."""
    return f"1 in {one_in_x:,} ({count:,} per 96M)"


def rewrite_odds_header(path: Path, one_in_x: int, count: int) -> bool:
    """Replace the `Odds: ...` line inside the header block. Returns True if
    the file changed. If no Odds line exists, inserts one after the Tier line."""
    text = path.read_text()
    value = odds_string(one_in_x, count)
    new_line = f"Odds: {value}"
    if _ODDS_RE.search(text):
        new_text = _ODDS_RE.sub(lambda m: m.group(1) + value, text, count=1)
    else:
        # insert after Tier: line if present, else give up quietly
        tier_re = re.compile(r"^(\s*Tier:.*)$", re.MULTILINE)
        m = tier_re.search(text)
        if not m:
            return False
        new_text = text[:m.end()] + "\n" + m.group(0)[:len(m.group(0)) - len(m.group(0).lstrip())] + new_line + text[m.end():]
    if new_text != text:
        path.write_text(new_text)
        return True
    return False


def find_pattern_file(eng, name):
    """Locate the .lua file that will SHIP for Pattern: <name>.

    IMPORTANT: do NOT trust the engine's resolved file_path -- a gitignored
    patterns/user/ shadow can SHADOW a shipped pattern (same internal name), and
    the engine resolves to the user copy. Baking there writes to an untracked
    file while the shipped copy (core/Nicks/Green Guide/...) gets nothing. So we
    scan for every file declaring the name and PREFER a non-user/ copy; fall back
    to a user/ file only when that's the only copy (e.g. 1959, a pure user
    pattern the web vendors directly)."""
    name_re = re.compile(rf"^\s*Pattern:\s*{re.escape(name)}\s*$", re.MULTILINE)
    matches = [f for f in (PROJECT_ROOT / "patterns").rglob("*.lua")
               if name_re.search(f.read_text(errors="ignore"))]
    if not matches:
        return None
    non_user = [f for f in matches if "user" not in f.parts]
    return non_user[0] if non_user else matches[0]


def _bake_from_report(report_path: Path):
    """Rewrite each pattern's Odds: header from a saved report (.json or .csv).
    No sweep -- uses the counts computed earlier. Takes seconds."""
    from pattern_engine_v3 import PatternEngineV3

    if report_path.suffix == ".json":
        rows = json.loads(report_path.read_text())
    else:
        with report_path.open(newline="") as f:
            rows = list(csv.DictReader(f))

    eng = PatternEngineV3()
    changed = skipped = 0
    expected = []   # (name, one_in_x, count) rows that SHOULD end up in a ship file
    missing = []    # expected rows whose ship file couldn't be found/updated
    for r in rows:
        name = r["name"]
        one_in_x = r.get("one_in_x")
        one_in_x = int(one_in_x) if one_in_x not in (None, "", "None") else None
        count = int(r["count"]) if r.get("count") not in (None, "", "None") else 0
        if not one_in_x or name in HARD_SKIP:
            skipped += 1
            continue
        expected.append((name, one_in_x, count))
        f = find_pattern_file(eng, name)
        if f and rewrite_odds_header(f, one_in_x, count):
            changed += 1
            print(f"  {name:32} -> {f.relative_to(PROJECT_ROOT)}")
        elif not (f and _header_has(f, odds_string(one_in_x, count))):
            missing.append(name)  # not found, or found but not carrying the value

    total = len(expected)
    print(f"\nbaked {changed} .lua files; {skipped} skipped (0 matches / hand-set).")
    # Verify EVERY expected pattern's ship file actually carries its computed odds
    # (a silently-skipped or shadowed file is the failure mode -- catch it here).
    verified = 0
    for name, one_in_x, count in expected:
        f = find_pattern_file(eng, name)
        if f and _header_has(f, odds_string(one_in_x, count)):
            verified += 1
        elif name not in missing:
            missing.append(name)
    print(f"verified {verified} of {total} shipped odds present in ship files.")
    if missing:
        print("!! MISSING (odds not in the shipping file -- CHECK for a user/ shadow):")
        for name in missing:
            print(f"     {name}")
    else:
        print("all expected odds confirmed in their shipping files.")


def _header_has(path: Path, value: str) -> bool:
    """True if the file's Odds: header currently equals `value`."""
    m = _ODDS_RE.search(path.read_text(errors="ignore"))
    return bool(m) and m.group(0).split("Odds:", 1)[1].strip() == value


# --- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min", type=int, default=DEFAULT_MIN)
    ap.add_argument("--max", type=int, default=DEFAULT_MAX)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 2))
    ap.add_argument("--chunk", type=int, default=2_000_000, help="serials per work unit")
    ap.add_argument("--all", action="store_true", help="all installed patterns, not just shipped")
    ap.add_argument("--only", type=str, default=None, help="comma-separated pattern names (sanity checks)")
    ap.add_argument("--out", type=str, default=None, help="write CSV report here (also writes .json)")
    ap.add_argument("--write-headers", action="store_true", help="bake Odds: into the .lua files (default: dry run)")
    ap.add_argument("--bake-from", type=str, default=None,
                    help="skip the sweep: bake Odds: into .lua files from a saved report (.json or .csv). Seconds, not hours.")
    args = ap.parse_args()

    # Fast path: bake straight from a saved report -- no recompute.
    if args.bake_from:
        _bake_from_report(Path(args.bake_from).expanduser())
        return

    only = [s.strip() for s in args.only.split(",")] if args.only else None
    targets, skipped, eng = load_targets(shipped_only=not args.all, only=only)

    print(f"range [{args.min:,}, {args.max:,}]  size={args.max - args.min + 1:,}")
    print(f"targets: {len(targets)} patterns   skipped: {len(skipped)}")
    for name, reason in skipped:
        print(f"  skip {name}: {reason}")
    if not targets:
        print("nothing to compute")
        return

    # build chunk list
    chunks = []
    n = args.min
    while n <= args.max:
        end = min(n + args.chunk - 1, args.max)
        chunks.append((n, end))
        n = end + 1

    scripts = [(name, script) for name, script in targets]
    names = [name for name, _ in targets]
    totals = [0] * len(names)

    t0 = time.time()
    done_serials = 0
    total_serials = args.max - args.min + 1
    with Pool(processes=args.workers, initializer=_init, initargs=(scripts,)) as pool:
        for counts in pool.imap_unordered(_count_chunk, chunks):
            for i, c in enumerate(counts):
                totals[i] += c
            # progress: chunks complete out of order, so track by summing a proxy
            done_serials += 0  # updated below via chunk sizes is not order-safe; use chunk count
            elapsed = time.time() - t0
    elapsed = time.time() - t0
    print(f"done in {elapsed/60:.1f} min ({args.workers} workers)")

    # assemble results
    size = args.max - args.min + 1
    rows = []
    for name, count in zip(names, totals):
        one_in_x = int(round(size / count)) if count else None
        info = eng.lua_patterns[name]
        rows.append({
            "name": name,
            # effective display name the app actually shows (label override ->
            # DisplayName header -> derived friendly name), never blank
            "display": eng.get_pattern_info(name).get("display_name", name),
            "count": count,
            "one_in_x": one_in_x,
            "rarity": (odds_string(one_in_x, count) if one_in_x
                       else "cannot occur on a printed note"),
            "derived_tier": derive_tier(one_in_x),
            "current_tier": getattr(info, "tier", None),
        })
    rows.sort(key=lambda r: (r["count"] if r["count"] else float("inf")))

    # print report
    print(f"\n{'PATTERN':32} {'COUNT':>10} {'RARITY':>18} {'tier':>4} {'was':>4}")
    for r in rows:
        print(f"{r['name']:32} {r['count']:>10,} {r['rarity']:>18} {r['derived_tier']:>4} {str(r['current_tier']):>4}")

    if args.out:
        out = Path(args.out).expanduser()
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        out.with_suffix(".json").write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {out} and {out.with_suffix('.json')}")

    if args.write_headers:
        print("\nbaking Odds: headers ...")
        changed = 0
        for r in rows:
            if not r["one_in_x"] or r["name"] in HARD_SKIP:
                continue
            f = find_pattern_file(eng, r["name"])
            if f and rewrite_odds_header(f, r["one_in_x"], r["count"]):
                changed += 1
        print(f"updated {changed} .lua files")
    else:
        print("\n(dry run -- pass --write-headers to bake Odds: into the .lua files)")


if __name__ == "__main__":
    main()
