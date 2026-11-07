"""Pick which queued live scans go into the next processing chunk.

Live (chunked) processing stages small groups of settled scans into part_NN
folders. A chunk must never split a bill's front from its back.

When the files use the `_b` back-suffix convention (FastFoto_00059.jpg +
FastFoto_00059_b.jpg -- what the in-app scanner writes), group by bill number,
not by count: a single missing back used to shift every later chunk by one
(front N went in with back N-1). A bill still missing a side is held back until
a few later bills have arrived complete, then sent on as-is so one lost side
can't stall the strap.

Other naming schemes (plain sequential numbers from a scanner's own software)
keep the old even-count cut, which is all they can support.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

# Complete bills that must arrive after an incomplete one before we stop
# waiting for its missing side.
ORPHAN_AFTER = 3


def _bill_key(p: Path):
    stem = p.stem
    if stem.lower().endswith("_b"):
        return stem[:-2].lower(), "B"
    return stem.lower(), "A"


def select_live_chunk(pending: Sequence[Path], flush: bool = False) -> List[Path]:
    """Files from `pending` (arrival order) to process next. `flush` = Stop
    was pressed: take everything."""
    pending = list(pending)
    if flush:
        return pending
    if not any(p.stem.lower().endswith("_b") for p in pending):
        take = len(pending) - (len(pending) % 2)
        return pending[:take] if take >= 2 else []

    order: List[str] = []
    groups = {}
    for p in pending:
        key, side = _bill_key(p)
        if key not in groups:
            groups[key] = {}
            order.append(key)
        groups[key][side] = p

    def complete(k):
        return "A" in groups[k] and "B" in groups[k]

    chosen: List[str] = []
    for i, k in enumerate(order):
        if complete(k):
            chosen.append(k)
            continue
        later_complete = sum(1 for k2 in order[i + 1:] if complete(k2))
        if later_complete >= ORPHAN_AFTER:
            chosen.append(k)          # its other side isn't coming
            continue
        break                          # wait for this bill's other side
    out: List[Path] = []
    for k in chosen:
        g = groups[k]
        out.extend(g[s] for s in ("A", "B") if s in g)
    return out
