#!/usr/bin/env python3
"""
eBay SOLD-order demand analysis (Phase B, from FIL's own store exports).

The live Browse-API harvester (ebay_harvester.py) gives ASKING prices on ACTIVE
listings. This tool reads FIL's eBay **order exports** ("All Orders YYYY.csv")
— which are REALIZED SALES — and produces the same per-pattern report, but with
what notes actually SOLD FOR. That is the true demand signal the memory flagged
as hard to get via API (Marketplace Insights is restricted); FIL's own store
data sidesteps it entirely.

It reuses ebay_harvester's title-mining vocabulary and aggregation, so the two
sources are directly comparable (asking vs realized per pattern family). It
also surfaces FIL's OWN naming vocabulary — the exact phrases a domain-expert
seller uses — which feeds the Essentials naming question.

PRIVACY: the order CSVs contain buyer PII (names, emails, addresses). This tool
reads ONLY the Item Title / Sold For / Sale Date / Quantity columns and writes
ONLY those (plus derived serial/pattern) to its outputs. No buyer data is read
or emitted. Do not commit the source CSVs.

Usage:
    python3 tools/ebay_orders.py                       # ~/Downloads/All Orders*.csv
    python3 tools/ebay_orders.py path/to/orders.csv ...
    python3 tools/ebay_orders.py --out ebay_sold/      # write report + CSVs

Dependency-free (Python standard library only).
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Reuse the harvester's extraction + aggregation so sources are comparable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ebay_harvester as eh  # noqa: E402

DEFAULT_GLOB = "~/Downloads/All Orders*.csv"

_PRICE_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
_DENOM_RE = [
    (1, re.compile(r"one\s*dollar|\$1\b|\b1\s*dollar", re.I)),
    (2, re.compile(r"two\s*dollar|\$2\b|\b2\s*dollar", re.I)),
    (5, re.compile(r"five\s*dollar|\$5\b|\b5\s*dollar", re.I)),
    (10, re.compile(r"ten\s*dollar|\$10\b|\b10\s*dollar", re.I)),
    (20, re.compile(r"twenty\s*dollar|\$20\b|\b20\s*dollar", re.I)),
    (100, re.compile(r"hundred\s*dollar|\$100\b", re.I)),
]
# Non-serial supply items FIL also sells (straps, holders) — not fancy notes.
_SUPPLY_RE = re.compile(
    r"strap|holder|sleeve|album|binder|currency\s*strap|band(s)?\b", re.I)


def parse_price(raw):
    if not raw:
        return None
    m = _PRICE_RE.search(raw.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def detect_denom(title):
    for denom, rx in _DENOM_RE:
        if rx.search(title):
            return denom
    return None


def load_orders(paths):
    """Read order CSVs -> list of normalized 'item' dicts (harvester schema)."""
    items = []
    for p in paths:
        p = Path(p).expanduser()
        if not p.exists():
            print(f"  ! not found: {p}", file=sys.stderr)
            continue
        label = f"sold:{p.stem}"
        with p.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        # Header is the first row whose first cell is 'Sales Record Number'.
        hdr_idx = next((i for i, r in enumerate(rows)
                        if r and r[0].strip() == "Sales Record Number"), None)
        if hdr_idx is None:
            print(f"  ! no header row in {p.name}", file=sys.stderr)
            continue
        idx = {h: j for j, h in enumerate(rows[hdr_idx])}
        try:
            c_title, c_sold = idx["Item Title"], idx["Sold For"]
            c_date, c_qty = idx["Sale Date"], idx["Quantity"]
        except KeyError as e:
            print(f"  ! missing column {e} in {p.name}", file=sys.stderr)
            continue
        n = 0
        for r in rows[hdr_idx + 1:]:
            if len(r) <= c_title or not r[c_title].strip():
                continue
            title = r[c_title].strip()
            items.append({
                "itemId": None,  # each row is a distinct sale; don't dedup
                "title": title,
                "price": {"value": parse_price(r[c_sold]), "currency": "USD"},
                "itemWebUrl": "",
                "_query": label,
                "_sale_date": r[c_date].strip() if len(r) > c_date else "",
                "_qty": r[c_qty].strip() if len(r) > c_qty else "",
                "_denom": detect_denom(title),
                "_is_supply": bool(_SUPPLY_RE.search(title)),
            })
            n += 1
        print(f"  {p.name}: {n} sold orders")
    return items


def summarize(items):
    """Per-family REALIZED sold stats + FIL vocabulary + coverage."""
    note_items = [it for it in items if not it["_is_supply"]]
    supply_items = [it for it in items if it["_is_supply"]]

    stats, rows, vocab = eh.build_report(note_items)

    # Which sold notes matched no known family (candidate vocab gaps).
    unmatched = [it["title"] for it in note_items
                 if not eh.match_patterns(it["title"])]

    # Realized-price sanity: overall sold-price distribution.
    all_prices = [it["price"]["value"] for it in note_items
                  if it["price"]["value"] is not None]
    denoms = Counter(it["_denom"] for it in note_items)
    return {
        "stats": stats, "rows": rows, "vocab": vocab,
        "n_note_sales": len(note_items),
        "n_supply_sales": len(supply_items),
        "unmatched": unmatched,
        "all_prices": all_prices,
        "denoms": denoms,
    }


def print_report(res):
    stats = res["stats"]
    print()
    print("=" * 74)
    print(f"REALIZED SOLD DATA — {res['n_note_sales']} note sales "
          f"({res['n_supply_sales']} supply/strap sales excluded)")
    print("=" * 74)
    if res["all_prices"]:
        p = res["all_prices"]
        print(f"Overall sold price: median ${statistics.median(p):.2f}  "
              f"mean ${statistics.mean(p):.2f}  "
              f"min ${min(p):.2f}  max ${max(p):.2f}  (n={len(p)})")
    denom_str = ", ".join(f"${d}: {c}" for d, c in
                          sorted(res["denoms"].items(),
                                 key=lambda kv: -kv[1]) if d)
    if denom_str:
        print(f"By denomination: {denom_str}")
    print()

    def money(v):
        return f"${v:,.2f}" if v is not None else "  -  "

    order = sorted(stats.items(), key=lambda kv: kv[1]["supply"], reverse=True)
    hdr = (f"{'Pattern family':<22}{'Sold':>6}{'Serials':>8}"
           f"{'Median':>10}{'Mean':>10}{'Max':>10}")
    print(hdr)
    print("-" * len(hdr))
    for fam, s in order:
        print(f"{fam:<22}{s['supply']:>6}{s['distinct_serials']:>8}"
              f"{money(s['price_median']):>10}{money(s['price_mean']):>10}"
              f"{money(s['price_max']):>10}")

    print()
    print("FIL's naming vocabulary (phrase -> sold listings):")
    for phrase, n in res["vocab"].most_common(25):
        print(f"  {n:>4}  {phrase}")

    if res["unmatched"]:
        print(f"\n{len(res['unmatched'])} sold notes matched NO known family "
              "(vocabulary gaps — candidates to add):")
        for t in res["unmatched"][:20]:
            print(f"    · {t[:72]}")


def write_outputs(out_dir, res):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sold_report.json").write_text(json.dumps({
        "patterns": res["stats"],
        "vocabulary": dict(res["vocab"].most_common()),
        "n_note_sales": res["n_note_sales"],
        "n_supply_sales": res["n_supply_sales"],
        "unmatched_titles": res["unmatched"],
    }, indent=2))

    with (out / "sold_patterns.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pattern", "sold", "distinct_serials", "priced",
                    "price_min", "price_median", "price_mean", "price_max"])
        for fam, s in sorted(res["stats"].items(),
                             key=lambda kv: kv[1]["supply"], reverse=True):
            w.writerow([fam, s["supply"], s["distinct_serials"],
                        s["priced_listings"], s["price_min"],
                        s["price_median"], s["price_mean"], s["price_max"]])

    # PII-free listing rows only.
    with (out / "sold_listings.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "title", "sold_for", "serial", "patterns"])
        for row in res["rows"]:
            w.writerow([row["query"], row["title"], row["price"],
                        row["serial"], row["patterns"]])
    print(f"\nWrote sold_report.json, sold_patterns.csv, sold_listings.csv "
          f"-> {out}/")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Analyze FIL's eBay sold-order exports for realized "
                    "per-pattern demand.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*",
                    help=f"Order CSVs. Default: {DEFAULT_GLOB}")
    ap.add_argument("--out", default=None, help="Directory for report + CSVs.")
    args = ap.parse_args(argv)

    paths = args.files or glob.glob(os.path.expanduser(DEFAULT_GLOB))
    if not paths:
        ap.error(f"No order CSVs given and none found at {DEFAULT_GLOB}")
    print(f"Loading {len(paths)} order file(s)...")
    items = load_orders(paths)
    if not items:
        ap.error("No sold orders parsed.")
    res = summarize(items)
    print_report(res)
    if args.out:
        write_outputs(args.out, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
