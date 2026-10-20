"""Fast, approximate serial rarity for a single Lua pattern.

The exact tool (tools/compute_rarity.py) brute-forces all 96M printed serials
with the real match() -- correct, but ~minutes and core-heavy, so not something
to run live in the Pattern Wizard. This module samples a random subset instead:
compile the pattern's match() once, run it over N random serials, and scale the
hit rate to a "per 96M" figure. For a common pattern a 200K sample lands within
~1% in under a second; for a very rare one a sample sees too few hits to be
trustworthy, so we flag that and defer to the exact tool.

Only DIGIT-DETERMINISTIC patterns can be estimated this way (match depends only
on the 8 digits). Patterns that read image metadata, a DataFile, or the current
date cannot -- same exclusions tools/compute_rarity.py makes -- so estimate()
reports estimable=False with a reason instead of guessing.
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Optional

RANGE_MAX = 96_000_000  # printed serial range [1, 96M]; top ~4M of a 10^8 block unprinted

# Same wrapper tools/compute_rarity.py uses: capture match() in a local and
# return a closure yielding a plain bool (avoids per-call table access).
_WRAP = ("\nlocal __m = match; return function(c) local r = __m(c); "
         "return (r and r.matched) and true or false end")

_DATE_TOKENS = ("current_year", "current_month", "current_day")


def can_estimate(script: str) -> tuple[bool, str]:
    """(True, "") if the pattern's match depends only on the 8 digits, else
    (False, reason). Mirrors compute_rarity's skip categories."""
    if re.search(r"\bDataFile\s*:", script) or "ctx.data" in script:
        return False, "it reads a data file"
    if "ctx.metadata" in script:
        return False, "it depends on image metadata (e.g. gas pump / seal shift)"
    if any(tok in script for tok in _DATE_TOKENS):
        return False, "its odds depend on the current date"
    return True, ""


def _compile(script: str, helpers_path: Optional[Path]):
    """Compile the pattern's match() once and return (match_fn, build_ctx).

    match_fn(ctx) -> bool; build_ctx(n) -> a Lua ctx table for serial n. Shared
    by the sampled estimate and the exact sweep so they agree bit-for-bit."""
    import lupa
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    lua.execute("function log(...) end")
    if helpers_path is None:
        from dollar_detective.resource_path import app_base
        helpers_path = app_base() / "patterns" / "lib" / "helpers.lua"
    if helpers_path and Path(helpers_path).exists():
        lua.execute(Path(helpers_path).read_text(encoding="utf-8"))
    fn = lua.execute(script + _WRAP)
    build = lua.eval(
        "function(n)"
        "  local s = string.format('%08d', n)"
        "  local dl = {}"
        "  for i=1,8 do dl[i] = tonumber(string.sub(s,i,i)) end"
        "  return {digits=s, full_serial=s, digit_list=dl, metadata={}}"
        "end"
    )
    return fn, build


def estimate(script: str, sample_size: int = 200_000, seed: Optional[int] = None,
             helpers_path: Optional[Path] = None) -> dict:
    """Estimate how often `script` matches, over a random sample of serials.

    Returns a dict:
      {estimable: False, reason: str}                      -- not digit-only
      {estimable: True, hits, sample_size, per_96m,
       one_in_x, reliable: bool}                            -- an estimate
      {estimable: True, error: str}                         -- failed to run

    `reliable` is False when the sample saw too few hits for the figure to mean
    much (the pattern is rare enough that the exact tool should be used).
    """
    ok, reason = can_estimate(script)
    if not ok:
        return {"estimable": False, "reason": reason}

    try:
        fn, build = _compile(script, helpers_path)
    except Exception as e:  # compile/load failure -- surface, don't crash the wizard
        return {"estimable": True, "error": str(e)}

    rng = random.Random(seed)
    hits = 0
    try:
        for _ in range(sample_size):
            if fn(build(rng.randint(1, RANGE_MAX))):
                hits += 1
    except Exception as e:
        return {"estimable": True, "error": str(e)}

    per_96m = round(hits / sample_size * RANGE_MAX) if sample_size else 0
    one_in_x = round(RANGE_MAX / per_96m) if per_96m else None
    # Need a handful of hits before the extrapolation means anything.
    reliable = hits >= 30
    return {
        "estimable": True,
        "hits": hits,
        "sample_size": sample_size,
        "per_96m": per_96m,
        "one_in_x": one_in_x,
        "reliable": reliable,
    }


def sweep_exact(script: str, helpers_path: Optional[Path] = None,
                cancel=None, progress=None) -> dict:
    """Exact match count over the whole printed range [1, 96M], single process.

    This is the same ground-truth brute force as tools/compute_rarity.py, but
    in-process so it works in the shipped app (no venv / tool script). ~5 min on
    one core, so callers run it on a background thread. `cancel()` -> True aborts
    (checked periodically); `progress(done, total)` is called occasionally.

    Returns {estimable: False, reason} for a non-digit pattern, {cancelled: True},
    {error: str}, or {exact: True, count, one_in_x, per_96m}."""
    ok, reason = can_estimate(script)
    if not ok:
        return {"estimable": False, "reason": reason}
    try:
        fn, build = _compile(script, helpers_path)
    except Exception as e:
        return {"estimable": True, "error": str(e)}

    count = 0
    next_progress = 2_000_000
    try:
        for n in range(1, RANGE_MAX + 1):
            if fn(build(n)):
                count += 1
            if (n & 0xFFFF) == 0:  # ~every 65k serials: cheap cancel/progress check
                if cancel and cancel():
                    return {"estimable": True, "cancelled": True}
                if progress and n >= next_progress:
                    progress(n, RANGE_MAX)
                    next_progress += 2_000_000
    except Exception as e:
        return {"estimable": True, "error": str(e)}

    one_in_x = round(RANGE_MAX / count) if count else None
    return {"estimable": True, "exact": True, "count": count,
            "one_in_x": one_in_x, "per_96m": count}


def odds_string(one_in_x: int, count: int) -> str:
    """The Odds: header format (matches tools/compute_rarity.py verbatim so the
    desktop and dd-web render identically)."""
    return f"1 in {one_in_x:,} ({count:,} per 96M)"


_ODDS_RE = re.compile(r"^(\s*Odds:\s*).*$", re.MULTILINE)
_TIER_RE = re.compile(r"^(\s*)Tier:.*$", re.MULTILINE)


def write_odds_header(path, one_in_x: int, count: int) -> bool:
    """Replace the Odds: line in a pattern .lua header, or insert one after the
    Tier: line if absent (newly-saved patterns have no Odds line). Returns True
    if the file changed."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    value = odds_string(one_in_x, count)
    if _ODDS_RE.search(text):
        new = _ODDS_RE.sub(lambda m: m.group(1) + value, text, count=1)
    else:
        m = _TIER_RE.search(text)
        if not m:
            return False
        indent = m.group(1)
        new = text[:m.end()] + f"\n{indent}Odds: {value}" + text[m.end():]
    if new != text:
        p.write_text(new, encoding="utf-8")
        return True
    return False


def format_estimate(result: dict) -> str:
    """One-line human string for the wizard label."""
    if not result.get("estimable", False):
        return f"Odds: set manually ({result.get('reason', 'not auto-computable')})"
    if result.get("error"):
        return "Estimated odds: (couldn't run estimate)"
    hits = result.get("hits", 0)
    n = result.get("sample_size", 0)
    if hits == 0:
        return (f"Estimated odds: rarer than ~1 in {n:,} "
                f"(0 hits in sample — use exact compute)")
    one_in_x = result.get("one_in_x")
    per_96m = result.get("per_96m", 0)
    base = f"Estimated odds: ~1 in {one_in_x:,} ({per_96m:,} per 96M)"
    if not result.get("reliable", True):
        return base + " — rough, few sample hits; confirm with exact compute"
    return base + f" (estimated from {n:,} sample)"
