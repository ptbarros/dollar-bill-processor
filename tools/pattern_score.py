#!/usr/bin/env python3
"""
Essentials pattern SCORING — join demand x keep x rarity into one ranked table.

This is the decision layer the eBay work feeds. It joins three axes per pattern:

  1. DEMAND   — from eBay sold orders (ebay_orders.py -> sold_report.json:
                realized median price + units sold) and/or the live Browse-API
                harvester (ebay_harvester.py -> report.json: asking price +
                supply). eBay families are mapped onto internal patterns using
                the SAME title-mining vocabulary, so the join is consistent.
  2. KEEP     — from FIL's ledger: per-pattern hits + how many he actually kept
                (cropped). Source = a populated ledger.db (--ledger) OR an
                analyze_scans keep CSV (--keep-csv). Optional.
  3. RARITY   — each pattern's hand-authored Tier (1 = rarest ... 10 = common),
                read from the pattern engine. Inverted into a 0-1 rarity score.

Each axis is normalized 0-1 (log-scaled for the skewed price/volume axes) and
combined with tunable weights (demand-leaning by default, per the project's
"weight the core toward demand" guidance). Missing axes are dropped and the
remaining weights renormalized, so it still runs with demand+rarity only (e.g.
before FIL's ledger is available on this machine).

Output is a ranked table + CSV/JSON, flagged with current Essentials membership
so you can see what the score would ADD or DROP vs the shipped 86-pattern set.
The composite is a transparent, re-weightable aid — NOT a black-box verdict;
the raw axes are always shown so FIL can arbitrate.

Usage:
    # demand + rarity (no keep data needed):
    python3 tools/pattern_score.py --sold ebay_sold/sold_report.json --out scores/
    # full three-axis on FIL's machine:
    python3 tools/pattern_score.py --sold sold_report.json --ledger ledger.db \
        --asking ebay_report/report.json --out scores/

Dependency-free (Python standard library only).
"""

import argparse
import csv
import json
import math
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ebay_harvester as eh  # noqa: E402  (shared title-mining vocabulary)

DEFAULT_WEIGHTS = {"value": 0.30, "volume": 0.25, "keep": 0.30, "rarity": 0.15}


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_patterns():
    """{internal_name: {display, tier}} for all installed patterns."""
    from pattern_engine_v3 import PatternEngineV3
    eng = PatternEngineV3()
    out = {}
    for name, info in eng.lua_patterns.items():
        disp = getattr(info, "display_name", "") or name
        out[name] = {"display": disp, "tier": int(getattr(info, "tier", 10))}
    return out


def load_demand(sold_path, asking_path):
    """Per eBay-family demand: {family: {sold, price_median, price_max,
    asking_supply, asking_median}}."""
    fam = {}

    def bucket(f):
        return fam.setdefault(f, {"sold": 0, "price_median": None,
                                  "price_max": None, "asking_supply": 0,
                                  "asking_median": None})

    if sold_path:
        d = json.loads(Path(sold_path).read_text())
        for f, s in d.get("patterns", {}).items():
            b = bucket(f)
            b["sold"] = s.get("supply", 0)
            b["price_median"] = s.get("price_median")
            b["price_max"] = s.get("price_max")
    if asking_path:
        d = json.loads(Path(asking_path).read_text())
        for f, s in d.get("patterns", {}).items():
            b = bucket(f)
            b["asking_supply"] = s.get("supply", 0)
            b["asking_median"] = s.get("price_median")
    return fam


def load_keep_from_ledger(ledger_path):
    """{internal_name: {hits, kept}} from a populated ledger.db."""
    con = sqlite3.connect(str(ledger_path))
    tables = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if "bill_patterns" not in tables:
        print(f"  ! {ledger_path} has no bill_patterns table (unpopulated) — "
              "skipping keep data.", file=sys.stderr)
        return {}
    keep = {}
    for pat, hits in con.execute(
            "SELECT pattern, COUNT(*) FROM bill_patterns GROUP BY pattern"):
        keep[pat] = {"hits": hits, "kept": 0}
    for pat, kept in con.execute(
            "SELECT bp.pattern, COUNT(*) FROM bill_patterns bp "
            "JOIN bills b ON b.id = bp.bill_id WHERE b.kept = 1 "
            "GROUP BY bp.pattern"):
        keep.setdefault(pat, {"hits": 0, "kept": 0})["kept"] = kept
    con.close()
    return keep


def load_keep_from_csv(csv_path):
    """{internal_name: {hits, kept}} from an analyze_scans keep CSV."""
    keep = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = row.get("pattern")
            if not name:
                continue
            keep[name] = {"hits": int(float(row.get("hits", 0) or 0)),
                          "kept": int(float(row.get("kept", 0) or 0))}
    return keep


# --------------------------------------------------------------------------- #
# Join + score
# --------------------------------------------------------------------------- #
def map_family(display, name, demand):
    """The eBay demand family for a pattern = the matched family with the most
    realized sales (falls back to asking supply). Returns (family, all_hits)."""
    text = f"{display} {name.replace('_', ' ')}"
    hits = [f for f in eh.match_patterns(text) if f in demand]
    if not hits:
        return None, []
    primary = max(hits, key=lambda f: (demand[f]["sold"],
                                       demand[f]["asking_supply"]))
    return primary, hits


def _norm_log(values):
    """Return a function mapping value -> 0..1 via log1p min-max over `values`."""
    logs = [math.log1p(v) for v in values if v is not None and v > 0]
    if not logs:
        return lambda v: 0.0
    lo, hi = min(logs), max(logs)
    span = (hi - lo) or 1.0
    return lambda v: 0.0 if (v is None or v <= 0) else \
        (math.log1p(v) - lo) / span


def build_rows(patterns, demand, keep, essentials):
    rows = []
    for name, meta in patterns.items():
        fam, all_fams = map_family(meta["display"], name, demand)
        d = demand.get(fam, {}) if fam else {}
        k = keep.get(name, {})
        hits, kept = k.get("hits", 0), k.get("kept", 0)
        rows.append({
            "name": name,
            "display": meta["display"],
            "tier": meta["tier"],
            "ebay_family": fam or "",
            "ebay_families": "; ".join(all_fams),
            "sold": d.get("sold", 0),
            "sold_median": d.get("price_median"),
            "sold_max": d.get("price_max"),
            "asking_supply": d.get("asking_supply", 0),
            "keep_hits": hits,
            "kept": kept,
            "keep_rate": (kept / hits) if hits else None,
            "in_essentials": name in essentials,
        })

    # Normalizers over the populated axes.
    val_n = _norm_log([r["sold_median"] for r in rows])
    vol_n = _norm_log([r["sold"] for r in rows])
    keep_n = _norm_log([r["kept"] for r in rows])
    have_keep = any(r["kept"] for r in rows)
    have_demand = any(r["sold"] or r["asking_supply"] for r in rows)

    for r in rows:
        r["s_value"] = round(val_n(r["sold_median"]), 3)
        r["s_volume"] = round(vol_n(r["sold"]), 3)
        r["s_keep"] = round(keep_n(r["kept"]), 3)
        r["s_rarity"] = round((11 - r["tier"]) / 10, 3)  # tier1 -> 1.0

        # Weight only the axes we actually have data for, then renormalize.
        w = dict(DEFAULT_WEIGHTS)
        if not have_demand:
            w["value"] = w["volume"] = 0.0
        if not have_keep:
            w["keep"] = 0.0
        tot = sum(w.values()) or 1.0
        r["score"] = round(
            (w["value"] * r["s_value"] + w["volume"] * r["s_volume"] +
             w["keep"] * r["s_keep"] + w["rarity"] * r["s_rarity"]) / tot, 4)
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows, have_keep, have_demand


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def money(v):
    return f"${v:,.2f}" if v is not None else "  -  "


def print_report(rows, have_keep, have_demand, top):
    axes = []
    if have_demand:
        axes.append("demand(eBay sold/asking)")
    if have_keep:
        axes.append("keep(ledger)")
    axes.append("rarity(tier)")
    print()
    print("=" * 92)
    print(f"PATTERN SCORE — {len(rows)} patterns · axes: {', '.join(axes)}")
    if not have_keep:
        print("  (no keep data — run with --ledger/--keep-csv on FIL's machine "
              "for the full three-axis score)")
    print("=" * 92)
    hdr = (f"{'#':>3} {'Pattern':<26}{'T':>2}{'Fam':<14}{'Sold':>5}"
           f"{'Med':>8}{'Kept':>6}{'K%':>6}  {'score':>6}  E")
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(rows[:top], 1):
        kr = f"{r['keep_rate']*100:.0f}%" if r["keep_rate"] is not None else "-"
        print(f"{i:>3} {r['display'][:26]:<26}{r['tier']:>2}"
              f"{r['ebay_family'][:13]:<14}{r['sold']:>5}"
              f"{money(r['sold_median']):>8}{r['kept']:>6}{kr:>6}  "
              f"{r['score']:>6.3f}  {'*' if r['in_essentials'] else ''}")


def essentials_delta(rows, have_keep, have_demand, protect_tier):
    """What the top scores would ADD to / DROP from the current Essentials set.

    Crown-jewel guard: patterns at/below `protect_tier` (the rarest) are NEVER
    dropped — they are so rare they appear in neither eBay sales nor the keep
    sample, so demand+keep are structurally blind to them and score them ~0.
    Rarity alone can't outweigh two zeroed axes, so we protect them by policy
    (see memory: include crown-jewels regardless of local frequency)."""
    in_ess = [r for r in rows if r["in_essentials"]]
    protected = [r for r in in_ess if r["tier"] <= protect_tier]
    protected_names = {r["name"] for r in protected}
    top_cut = len(in_ess)  # compare same-size cut
    top = set(r["name"] for r in rows[:top_cut])
    would_add = [r for r in rows[:top_cut]
                 if not r["in_essentials"] and r["tier"] > protect_tier]
    would_drop = [r for r in in_ess
                  if r["name"] not in top and r["name"] not in protected_names]
    print("\n" + "-" * 60)
    if not have_keep:
        print("CAVEAT: no keep axis — patterns whose value is KEEP-driven with")
        print("no eBay footprint (GAS_PUMP, Seal Shift, year/date stand-alones)")
        print("score low here for lack of data, NOT because they're weak. The")
        print("DROP list is unreliable until run with --ledger/--keep-csv.")
    if have_demand:
        print("NOTE: eBay demand is family-coarse (a dozen families over 389")
        print("patterns), so same-family variants inherit one demand and tie;")
        print("keep + tier are what separate them.")
    print(f"ESSENTIALS DELTA (top {top_cut} by score vs current {len(in_ess)}):")
    if protected:
        print(f"  PROTECTED {len(protected)} crown-jewels (tier <= "
              f"{protect_tier}, kept regardless of score): "
              + ", ".join(sorted(r["display"][:20] for r in protected))[:200])
    print(f"  score would ADD {len(would_add)} not in Essentials:")
    for r in would_add[:15]:
        print(f"    + {r['display'][:34]:<34} score {r['score']:.3f} "
              f"(sold {r['sold']}, tier {r['tier']})")
    print(f"  score would DROP {len(would_drop)} currently in Essentials "
          "(lowest-scoring members):")
    for r in sorted(would_drop, key=lambda r: r["score"])[:15]:
        print(f"    - {r['display'][:34]:<34} score {r['score']:.3f} "
              f"(sold {r['sold']}, kept {r['kept']}, tier {r['tier']})")


def write_outputs(out_dir, rows):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["name", "display", "tier", "ebay_family", "ebay_families", "sold",
            "sold_median", "sold_max", "asking_supply", "keep_hits", "kept",
            "keep_rate", "s_value", "s_volume", "s_keep", "s_rarity", "score",
            "in_essentials"]
    with (out / "pattern_scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    (out / "pattern_scores.json").write_text(json.dumps(rows, indent=2))
    print(f"\nWrote pattern_scores.csv, pattern_scores.json -> {out}/")


# --------------------------------------------------------------------------- #
# Proposed Essentials selection
# --------------------------------------------------------------------------- #
def add_reasons(r, args):
    """Positive evidence to ADD a pattern to Essentials — union across axes:
    the market wants it (and he keeps it) OR he keeps it OR it's a crown jewel.
    Demand is FAMILY-coarse so it may only promote a pattern with a real
    pattern-specific keep footprint (kept>=1), never add exotic variants he
    never encounters. Returns triggered reasons (empty => no add signal)."""
    reasons = []
    if r["tier"] <= args.include_tier:
        reasons.append("crown-jewel")
    if r["kept"] >= args.keep_vol_min:
        reasons.append("keep-volume")
    if r["keep_hits"] >= 3 and (r["keep_rate"] or 0) >= args.keep_rate_min:
        reasons.append("keep-conviction")
    if r["kept"] >= 1:
        if r["sold"] >= args.demand_sold_min:
            reasons.append("demand-volume")
        if (r["sold_median"] or 0) >= args.demand_price_min:
            reasons.append("demand-value")
    return reasons


def is_noise(r, args):
    """Positive evidence a CURRENT member is noise: it fires constantly yet is
    almost never kept. Absence from the sample is NOT noise — only a pattern
    with many hits and a near-zero keep rate qualifies to be dropped."""
    return (r["keep_hits"] >= args.noise_hits_min and
            (r["keep_rate"] or 0) <= args.noise_keep_max)


def emit_selection(rows, args, current_essentials):
    """Write an importable pattern-selection JSON.

    ADD-oriented: the current keep-validated set is the baseline. Data ADDS
    patterns with positive evidence and DROPS a current member only with
    positive noise evidence — never merely for being absent from the sample
    (crown jewels are too rare to appear, but must stay)."""
    from collections import Counter
    disp = {r["name"]: r["display"] for r in rows}
    included = set(current_essentials)
    add_why, reason_counts = {}, Counter()
    added, dropped = [], []

    for r in rows:
        why = add_reasons(r, args)
        r["_why"] = ";".join(why)
        if why:
            for w in why:
                reason_counts[w] += 1
            if r["name"] not in current_essentials:
                included.add(r["name"])
                added.append(r["name"])
                add_why[r["name"]] = why
        # Drop a current member only on positive noise evidence.
        if r["name"] in current_essentials and is_noise(r, args):
            included.discard(r["name"])
            dropped.append(r["name"])

    states = {r["name"]: (r["name"] in included) for r in rows}
    doc = {
        "format": "dollar-detective-pattern-selection",
        "version": 1,
        "name": "Essentials (proposed, data-scored)",
        "description": ("Data-scored Essentials proposal: current keep-validated "
                        "set + demand (eBay sold) & keep (real straps) adds, "
                        "crown jewels protected, only proven-noise drops. "
                        "Generated by tools/pattern_score.py — arbitrate before "
                        "shipping."),
        "library_states": {},
        "pattern_states": states,
    }
    Path(args.emit_selection).write_text(json.dumps(doc, indent=2))

    print("\n" + "=" * 60)
    print(f"PROPOSED ESSENTIALS — {len(included)} patterns "
          f"(current {len(current_essentials)}: +{len(added)} / -{len(dropped)})")
    print("=" * 60)
    print("ADD signals present (a pattern can trigger several):")
    for w, n in reason_counts.most_common():
        print(f"  {n:>4}  {w}")
    print(f"\n  ADDED ({len(added)}):")
    for name in sorted(added, key=lambda n: add_why[n]):
        print(f"    + {disp.get(name, name)} [{';'.join(add_why[name])}]")
    print(f"\n  DROPPED as noise ({len(dropped)}):")
    for name in sorted(dropped):
        r = next(x for x in rows if x["name"] == name)
        print(f"    - {disp.get(name, name)} (hits {r['keep_hits']}, "
              f"keep {r['keep_rate']*100:.0f}%)")
    if not dropped:
        print("    (none — no current member shows noise evidence)")
    print(f"\nWrote selection -> {args.emit_selection}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Score patterns by demand x keep x rarity.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sold", help="ebay_orders.py sold_report.json (realized).")
    ap.add_argument("--asking", help="ebay_harvester.py report.json (asking).")
    ap.add_argument("--ledger", help="Populated ledger.db for keep data.")
    ap.add_argument("--keep-csv", help="analyze_scans keep CSV for keep data.")
    ap.add_argument("--essentials", default="essentials_default.json",
                    help="Essentials selection JSON (for the * flag / delta).")
    ap.add_argument("--top", type=int, default=40, help="Rows to print.")
    ap.add_argument("--protect-tier", type=int, default=2,
                    help="Never drop patterns at/below this tier "
                         "(crown-jewel floor). Default 2.")
    ap.add_argument("--out", help="Directory for CSV/JSON.")
    # Proposed-selection inclusion thresholds (union rule; any one includes).
    ap.add_argument("--emit-selection", metavar="PATH",
                    help="Write an importable proposed Essentials JSON.")
    ap.add_argument("--include-tier", type=int, default=2,
                    help="ADD patterns at/below this tier (crown jewels). "
                         "Default 2.")
    ap.add_argument("--keep-vol-min", type=int, default=5,
                    help="ADD if kept >= this. Default 5.")
    ap.add_argument("--keep-rate-min", type=float, default=0.30,
                    help="ADD if hits>=3 and keep_rate >= this. Default .30.")
    ap.add_argument("--demand-sold-min", type=int, default=8,
                    help="ADD if kept>=1 and eBay family sold >= this. Def 8.")
    ap.add_argument("--demand-price-min", type=float, default=15.0,
                    help="ADD if kept>=1 and family median >= this $. Def 15.")
    ap.add_argument("--noise-hits-min", type=int, default=40,
                    help="DROP a current member only if hits >= this AND "
                         "keep-rate <= --noise-keep-max. Default 40.")
    ap.add_argument("--noise-keep-max", type=float, default=0.05,
                    help="Keep-rate ceiling for a noise drop. Default .05.")
    args = ap.parse_args(argv)

    if not (args.sold or args.asking):
        ap.error("Need at least --sold and/or --asking for the demand axis.")

    patterns = load_patterns()
    demand = load_demand(args.sold, args.asking)
    keep = {}
    if args.ledger:
        keep = load_keep_from_ledger(args.ledger)
    elif args.keep_csv:
        keep = load_keep_from_csv(args.keep_csv)

    essentials = set()
    ep = Path(args.essentials)
    if ep.exists():
        ps = json.loads(ep.read_text()).get("pattern_states", {})
        essentials = {k for k, v in ps.items() if v}

    rows, have_keep, have_demand = build_rows(patterns, demand, keep, essentials)
    print_report(rows, have_keep, have_demand, args.top)
    if essentials:
        essentials_delta(rows, have_keep, have_demand, args.protect_tier)
    if args.emit_selection:
        emit_selection(rows, args, essentials)
    if args.out:
        write_outputs(args.out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
