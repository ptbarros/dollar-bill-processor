"""Suggested line-3 annotations for labels.

The FIL hand-types a per-bill "feature" note on line 3 that spells out *what makes
the bill sellable* — the actual digits, their position, or the grouping. Most of
that is already computed by the pattern engine (the highlight/group-box geometry
of the match), so we can suggest it and let him accept or tweak, instead of him
typing it every time.

Derived from analysing his real label file (FINAL LABEL for Paul.docx):
  4 DIGIT LADDER  "6543  LEADING" / "2345"   <- the run digits + position
  HYBRID          "70 5858 70"               <- serial split by its group boxes
  (low run)       "NOT DUPLICATE 3.2M/22.65M"<- run size (future: from low_runs)

`suggest_annotation` takes the serial and the engine's matches and returns a
best-effort string (or "" when it has nothing confident to say).
"""
from typing import List, Optional


def _digits(serial: str) -> str:
    """The 8 numeric characters of a serial (drops the prefix/suffix letters)."""
    return "".join(c for c in (serial or "") if c.isdigit())


def _grouped(digits: str, boxes) -> str:
    """Split digits into space-separated segments at the group-box boundaries.

    e.g. digits "70585870" with boxes (0-1) and (6-7) -> "70 5858 70".
    """
    try:
        boxes = sorted(({"from": int(b.get("from", 0)), "to": int(b.get("to", 0))}
                        for b in boxes), key=lambda b: b["from"])
    except (TypeError, ValueError, AttributeError):
        return ""
    segs, i = [], 0
    for b in boxes:
        a, z = b["from"], b["to"]
        if not (0 <= a <= z < len(digits)):
            continue
        if i < a:
            segs.append(digits[i:a])   # ungrouped gap
        segs.append(digits[a:z + 1])   # the boxed group
        i = z + 1
    if i < len(digits):
        segs.append(digits[i:])
    segs = [s for s in segs if s]
    return " ".join(segs) if len(segs) > 1 else ""


# Low-run print-run size by pattern name (the buckets in low_runs.csv).
_LOW_RUN_SIZE = {"LOW_RUN_6M": "6.4M", "LOW_RUN_12M": "12.8M"}


def suggest_annotation(serial: str, matches, prefer_names: Optional[List[str]] = None,
                       known_names: Optional[List[str]] = None) -> str:
    """A suggested line-3 note for a bill, or "" if nothing confident applies.

    `matches` is the engine's classify() result (objects with .name and
    .group_boxes) — used for ladder/grouping geometry. `known_names` are the
    bill's ALREADY-matched pattern names from processing (needed for low-run,
    which depends on image metadata a bare re-classify doesn't have).
    `prefer_names` (the selected pattern(s)) are tried first.
    """
    digits = _digits(serial)
    matches = matches or []
    if len(digits) < 4 and not known_names:
        return ""

    ordered = list(matches)
    if prefer_names:
        pset = set(prefer_names)
        ordered = ([m for m in matches if getattr(m, "name", "") in pset] +
                   [m for m in matches if getattr(m, "name", "") not in pset])

    # 1. Ladder / consecutive run: the boxed run digits + where they sit.
    for m in ordered:
        name = (getattr(m, "name", "") or "").upper()
        boxes = getattr(m, "group_boxes", None) or []
        if "LADDER" in name and boxes:
            try:
                a, b = int(boxes[0].get("from")), int(boxes[0].get("to"))
            except (TypeError, ValueError):
                continue
            if 0 <= a <= b < len(digits):
                run = digits[a:b + 1]
                if a == 0:
                    return f"{run}  LEADING"
                if b == len(digits) - 1:
                    return f"{run}  TRAILING"
                return run

    # 2. Bookend / repeater grouping: show the serial split by its groups.
    for m in ordered:
        name = (getattr(m, "name", "") or "").upper()
        boxes = getattr(m, "group_boxes", None) or []
        if boxes and ("BOOKEND" in name or "REPEATER" in name):
            grouped = _grouped(digits, boxes)
            if grouped:
                return grouped

    # 3. Low run: the print-run size, from the bill's matched patterns.
    names = list(known_names or []) + [getattr(m, "name", "") for m in matches]
    for nm in names:
        size = _LOW_RUN_SIZE.get(nm)
        if size:
            return f"{size} run"

    return ""
