#!/usr/bin/env python3
"""
Parse low runs from a uspapermoney.io serial chart.

Implements the derivation documented in patterns/core/LOW_RUNS.md:
  - each district's serial count is followed month by month,
  - each month's increment is attributed to that month's facility (FW/DC),
  - the count is split at every implied 96,000,000 block cap,
  - a facility's slice of a single block equal to 6.4M or 12.8M is a low run,
  - a block still printing in the latest month is "open" (not a low run yet).

This is a maintenance helper, NOT wired into the app. Download the chart page
first (the site has no API), then run this against the saved HTML:

    # Series 2021 $1
    curl -sL -A Mozilla https://www.uspapermoney.io/serials/f2021_s.html -o f2021_s.html
    python tools/parse_low_runs.py f2021_s.html --series 2021

Output is candidate `low_runs.csv` rows plus a list of still-open blocks to
revisit later. Always eyeball the result against the chart before committing —
the whole point of low runs is not to trust a number you didn't check.
"""

import argparse
import html
import re
import sys

DISTRICTS = list("ABCDEFGHIJKL")
BLOCK_ORDER = "ABCDEFGHIJKLMNPQRSTUVWXYZ"  # suffix blocks skip the letter 'O'
CAP = 96_000_000                            # implied notes-per-block cap
LOW_RUN_SIZES = {6_400_000: "6.4", 12_800_000: "12.8"}

# A serial token inside a cell, e.g. "I 896 00000 A" -> district I, 89600000, block A
TOKEN_RE = re.compile(r"([A-L])\s+(\d{3})\s*(\d{5})\s+([A-Z])")


def _cum(block: str, serial: int) -> int:
    """Cumulative note index for a (block, within-block serial)."""
    return BLOCK_ORDER.index(block) * CAP + serial


def parse_chart(html_text: str):
    """Return (portions, last_month, final_month).

    portions:   {(district, block, facility): notes}
    last_month: {(district, block, facility): month_index}
    """
    rows = re.findall(r"<tr>(.*?)</tr>", html_text, re.S)

    # Per district, chronological list of (month_index, cum_end, facility).
    per_district = {d: [] for d in DISTRICTS}
    month_index = -1
    for row in rows:
        cells = re.findall(r"<td([^>]*)>(.*?)</td>", row, re.S)
        if len(cells) < 12:
            continue  # header / spacer / star-note rows
        month_index += 1
        for i, dist in enumerate(DISTRICTS):
            attrs, inner = cells[i]
            facility = "FW" if "fw" in attrs else "DC"
            text = html.unescape(re.sub(r"<[^>]+>", " ", inner)).replace("\xa0", " ")
            toks = TOKEN_RE.findall(text)
            if not toks:
                continue
            # The last serial in the cell is this month's high-water mark.
            _, g3, g5, block = toks[-1]
            per_district[dist].append((month_index, _cum(block, int(g3 + g5)), facility))
    final_month = month_index

    portions, last_month = {}, {}
    for dist in DISTRICTS:
        prev = 0
        for _m, cum, facility in per_district[dist]:
            if cum <= prev:
                prev = max(prev, cum)
                continue
            lo, hi = prev, cum
            x = lo
            while x < hi:                       # split the increment across 96M blocks
                blk_idx = x // CAP
                seg_hi = min(hi, (blk_idx + 1) * CAP)
                key = (dist, BLOCK_ORDER[blk_idx], facility)
                portions[key] = portions.get(key, 0) + (seg_hi - x)
                last_month[key] = _m
                x = seg_hi
            prev = cum
    return portions, last_month, final_month


def main():
    ap = argparse.ArgumentParser(description="Parse low runs from a uspapermoney.io serial chart.")
    ap.add_argument("html_file", help="Saved chart HTML (e.g. f2021_s.html)")
    ap.add_argument("--series", required=True, help="Series label for the CSV rows, e.g. 2021 or 2017A")
    args = ap.parse_args()

    try:
        with open(args.html_file, "r", encoding="utf-8", errors="replace") as f:
            html_text = f.read()
    except OSError as e:
        print(f"Could not read {args.html_file}: {e}", file=sys.stderr)
        return 1

    portions, last_month, final_month = parse_chart(html_text)

    complete, still_open = [], []
    for key in sorted(portions):
        size = portions[key]
        if size not in LOW_RUN_SIZES:
            continue
        dist, block, facility = key
        row = f"{args.series},{dist},{block},{facility},{LOW_RUN_SIZES[size]}"
        if last_month[key] == final_month:
            still_open.append(row)          # printing in the latest month -> not finished
        else:
            complete.append(row)

    print("# Candidate low_runs.csv rows (complete blocks):")
    print("series,district,block,facility,quantity")
    for r in complete:
        print(r)

    if still_open:
        print("\n# Still printing in the latest chart month -- do NOT add yet, revisit:")
        for r in still_open:
            print(f"#   {r}")

    print(f"\n# {len(complete)} complete low run(s), {len(still_open)} still open. "
          f"Verify against the chart before committing.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
