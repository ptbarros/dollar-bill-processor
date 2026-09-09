#!/usr/bin/env python3
"""
eBay fancy-serial demand harvester (Phase A).

Queries the eBay **Browse API** for ACTIVE fancy-serial banknote listings and
mines each listing TITLE for (a) the 8-digit serial and (b) the seller's
claimed pattern name, then aggregates per pattern family:
  - SUPPLY      how many active listings mention it
  - ASKING PRICE  min / median / mean / max of the listing prices
  - VOCABULARY    the exact words sellers use to name each pattern

This is the free, ToS-compliant *demand-axis* feed for the Essentials pattern
set (project memory: ebay-demand-data / core-pattern-consolidation-idea).
It is a DEMAND signal (skewed to common/easy patterns), NOT a rarity signal —
pair it with Tier (rarity) + FIL keep-data. Realized SOLD prices are a later
phase (Terapeak / Marketplace Insights), not obtainable here.

--- Getting a keyset (free) ---
  1. Create a free developer account at https://developer.ebay.com
  2. Make a *production* keyset (App ID / Cert ID = Client ID / Client Secret)
  3. Supply them via (highest precedence first):
       --client-id / --client-secret
       env EBAY_CLIENT_ID / EBAY_CLIENT_SECRET
       a JSON creds file (--creds, default ~/.dbp_ebay.json):
         {"client_id": "...", "client_secret": "..."}
The Browse API is free: 5,000 calls/day (raisable via eBay's free
"Application Growth Check"). One search page (up to 200 items) = one call.

Usage:
    python3 tools/ebay_harvester.py                         # default query set
    python3 tools/ebay_harvester.py -q "binary serial dollar bill"
    python3 tools/ebay_harvester.py --max-pages 3 --out ebay_report/
    python3 tools/ebay_harvester.py --sandbox              # hit eBay sandbox
    python3 tools/ebay_harvester.py --save-raw raw.json    # cache API results
    python3 tools/ebay_harvester.py --replay raw.json      # offline, no keys

Dependency-free (Python standard library only).
"""

import argparse
import base64
import csv
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
ENDPOINTS = {
    "production": {
        "oauth": "https://api.ebay.com/identity/v1/oauth/token",
        "browse": "https://api.ebay.com/buy/browse/v1/item_summary/search",
    },
    "sandbox": {
        "oauth": "https://api.sandbox.ebay.com/identity/v1/oauth/token",
        "browse": "https://api.sandbox.ebay.com/buy/browse/v1/item_summary/search",
    },
}
OAUTH_SCOPE = "https://api.ebay.com/oauth/api_scope"
MARKETPLACE = "EBAY_US"
MAX_LIMIT = 200  # Browse API max item_summary page size

# Default searches: broad fancy-serial nets + one per major pattern family so
# every family gets a fair shot at supply/price data (not just whatever "fancy
# serial number" happens to surface).
DEFAULT_QUERIES = [
    "fancy serial number dollar bill",
    "fancy serial number one dollar",
    "binary serial number dollar bill",
    "trinary serial number dollar bill",
    "radar serial number dollar bill",
    "repeater serial number dollar bill",
    "ladder serial number dollar bill",
    "solid serial number dollar bill",
    "low serial number dollar bill",
    "birthday serial number dollar bill",
    "star note fancy serial dollar",
]

# --------------------------------------------------------------------------- #
# Pattern vocabulary — seller phrase -> canonical family.
# Ordered MOST-SPECIFIC FIRST; the first phrase found in a title wins for that
# family, but a title can match several families (e.g. "binary radar").
# Canonical names lean on the Essentials primitives taxonomy.
# --------------------------------------------------------------------------- #
PATTERN_TERMS = [
    ("Super Radar", [r"super\s*radar"]),
    ("Radar / Palindrome", [r"\bradar\b", r"palindrome"]),
    ("Super Repeater", [r"super\s*repeater"]),
    ("Repeater", [r"\brepeater\b", r"\brepeaters\b"]),
    ("True Binary", [r"true\s*binary"]),
    ("Binary", [r"\bbinary\b"]),
    ("Trinary", [r"\btrinary\b", r"ternary"]),
    ("Quinary", [r"\bquinary\b"]),
    ("Flipper", [r"\bflipper\b", r"flip\s*flop"]),
    ("Ladder", [r"\bladder\b", r"ladders"]),
    ("Solid", [r"\bsolid\b", r"solids", r"\b7\s*of\s*a\s*kind\b", r"7oak",
               r"\bseven\s*of\s*a\s*kind\b"]),
    ("Of-a-kind (6+)", [r"\b6\s*of\s*a\s*kind\b", r"6oak",
                        r"\bsix\s*of\s*a\s*kind\b"]),
    ("Bookend", [r"book\s*end", r"bookends?"]),
    ("Rotator", [r"\brotator\b", r"\brotate"]),
    ("Consecutive / Run", [r"consecutive", r"in a row"]),
    ("Quad", [r"\bquad\b", r"quads", r"four\s*of\s*a\s*kind", r"4oak",
              r"two\s*pair"]),
    ("Trio / Triple", [r"\btrio\b", r"triple", r"three\s*of\s*a\s*kind",
                       r"3oak"]),
    ("Repdigit", [r"repdigit", r"rep\s*digit"]),
    ("Low Serial", [r"low\s*serial", r"low\s*number", r"low\s*#"]),
    ("High Serial", [r"high\s*serial", r"high\s*number"]),
    ("Odometer", [r"odometer"]),
    ("ZIP Code", [r"zip\s*code"]),
    ("Tombstone", [r"tombstone"]),
    ("Year / Birth Note", [r"year\s*note", r"birth\s*year", r"birth\s*anniv",
                           r"anniversary"]),
    ("Birthday / Date", [r"birthday", r"\bbirth\s*date", r"\bdate\s*note"]),
    ("Silver Certificate", [r"silver\s*certificate"]),
    ("Star Note", [r"star\s*note", r"\bstar\b"]),
    ("Fancy (generic)", [r"fancy\s*serial", r"fancy\s*number", r"fancy\s*#"]),
]
# Pre-compile.
_PATTERN_RE = [(name, [re.compile(p, re.IGNORECASE) for p in pats])
               for name, pats in PATTERN_TERMS]

# Serial forms in titles. Full FRN $1 form (letter, 8 digits, letter/star) is
# strongest; a bare 8-digit run is the fallback.
_SERIAL_FULL = re.compile(r"\b([A-L\*])\s?(\d{8})\s?([A-Z\*])\b")
_SERIAL_BARE = re.compile(r"\b(\d{8})\b")


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
def resolve_creds(args):
    """Return (client_id, client_secret) from args > env > creds file."""
    cid = args.client_id or os.environ.get("EBAY_CLIENT_ID")
    sec = args.client_secret or os.environ.get("EBAY_CLIENT_SECRET")
    if cid and sec:
        return cid, sec
    creds_path = Path(args.creds).expanduser()
    if creds_path.exists():
        try:
            data = json.loads(creds_path.read_text())
            cid = cid or data.get("client_id")
            sec = sec or data.get("client_secret")
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ! could not read creds file {creds_path}: {e}",
                  file=sys.stderr)
    return cid, sec


# --------------------------------------------------------------------------- #
# eBay API
# --------------------------------------------------------------------------- #
def get_app_token(client_id, client_secret, oauth_url):
    """Client-credentials OAuth2 flow -> application access token."""
    basic = base64.b64encode(
        f"{client_id}:{client_secret}".encode()).decode()
    body = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "scope": OAUTH_SCOPE,
    }).encode()
    req = urllib.request.Request(
        oauth_url, data=body, method="POST",
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise SystemExit(
            f"OAuth failed ({e.code}). Check your keyset.\n{detail}")
    return payload["access_token"]


def browse_search(token, query, browse_url, limit, offset, category=None,
                  marketplace=MARKETPLACE):
    """One Browse item_summary page. Returns the parsed JSON dict."""
    params = {"q": query, "limit": limit, "offset": offset}
    if category:
        params["category_ids"] = category
    url = f"{browse_url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": marketplace,
        "Content-Type": "application/json",
    })
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:  # rate limited: back off and retry
                wait = 2 ** attempt
                print(f"    rate-limited, waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            detail = e.read().decode(errors="replace")[:400]
            print(f"    ! search HTTP {e.code}: {detail}", file=sys.stderr)
            return {}
        except urllib.error.URLError as e:
            print(f"    ! network error: {e}", file=sys.stderr)
            return {}
    return {}


def harvest_live(queries, token, browse_url, max_pages, category, delay):
    """Fetch item summaries for every query. Returns list of raw item dicts,
    each tagged with the query that surfaced it."""
    items = []
    for q in queries:
        print(f"  query: {q!r}")
        for page in range(max_pages):
            offset = page * MAX_LIMIT
            data = browse_search(token, q, browse_url, MAX_LIMIT, offset,
                                  category)
            summaries = data.get("itemSummaries") or []
            for it in summaries:
                it["_query"] = q
            items.extend(summaries)
            total = data.get("total", 0)
            print(f"    page {page + 1}: {len(summaries)} items "
                  f"(total available: {total})")
            if len(summaries) < MAX_LIMIT or offset + MAX_LIMIT >= total:
                break
            if delay:
                time.sleep(delay)
    return items


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract_serial(title):
    """Best-effort serial from a title. Returns a string or ''."""
    m = _SERIAL_FULL.search(title)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}".upper()
    m = _SERIAL_BARE.search(title)
    return m.group(1) if m else ""


def match_patterns(title):
    """Return the list of canonical family names whose vocabulary appears."""
    found = []
    for name, regexes in _PATTERN_RE:
        if any(rx.search(title) for rx in regexes):
            found.append(name)
    return found


def item_price(item):
    """(value, currency) for a listing, from price or current bid. Nones ok."""
    for key in ("price", "currentBidPrice"):
        p = item.get(key)
        if isinstance(p, dict) and p.get("value") is not None:
            try:
                return float(p["value"]), p.get("currency", "USD")
            except (TypeError, ValueError):
                pass
    return None, None


def matched_vocab(title):
    """The literal phrase(s) a title used, for the vocabulary corpus."""
    hits = []
    for name, regexes in _PATTERN_RE:
        for rx in regexes:
            m = rx.search(title)
            if m:
                hits.append(m.group(0).lower().strip())
                break
    return hits


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def build_report(items):
    """Turn raw items into (per_pattern_stats, rows, vocab_counter)."""
    per = defaultdict(lambda: {"count": 0, "prices": [], "serials": set(),
                               "titles": []})
    vocab = Counter()
    rows = []
    seen_items = set()

    for it in items:
        item_id = it.get("itemId")
        if item_id and item_id in seen_items:
            continue  # same listing surfaced by two queries
        if item_id:
            seen_items.add(item_id)

        title = (it.get("title") or "").strip()
        if not title:
            continue
        serial = extract_serial(title)
        families = match_patterns(title)
        value, currency = item_price(it)
        for phrase in matched_vocab(title):
            vocab[phrase] += 1

        rows.append({
            "query": it.get("_query", ""),
            "item_id": item_id or "",
            "title": title,
            "price": value if value is not None else "",
            "currency": currency or "",
            "serial": serial,
            "patterns": "; ".join(families),
            "url": it.get("itemWebUrl", ""),
        })

        for fam in families:
            bucket = per[fam]
            bucket["count"] += 1
            if value is not None:
                bucket["prices"].append(value)
            if serial:
                bucket["serials"].add(serial)
            if len(bucket["titles"]) < 5:
                bucket["titles"].append(title)

    stats = {}
    for fam, b in per.items():
        prices = b["prices"]
        stats[fam] = {
            "supply": b["count"],
            "distinct_serials": len(b["serials"]),
            "price_min": min(prices) if prices else None,
            "price_median": statistics.median(prices) if prices else None,
            "price_mean": round(statistics.mean(prices), 2) if prices else None,
            "price_max": max(prices) if prices else None,
            "priced_listings": len(prices),
            "sample_titles": b["titles"],
        }
    return stats, rows, vocab


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def print_summary(stats, rows, vocab, n_items):
    print()
    print("=" * 72)
    print(f"HARVEST SUMMARY — {len(rows)} unique listings "
          f"(from {n_items} raw results)")
    print("=" * 72)
    if not stats:
        print("No pattern families matched. Try different --query terms.")
        return

    def money(v):
        return f"${v:,.2f}" if v is not None else "  -  "

    order = sorted(stats.items(), key=lambda kv: kv[1]["supply"], reverse=True)
    hdr = f"{'Pattern family':<22}{'Supply':>7}{'Serials':>8}" \
          f"{'Median':>11}{'Mean':>11}{'Range':>22}"
    print(hdr)
    print("-" * len(hdr))
    for fam, s in order:
        rng = f"{money(s['price_min'])}–{money(s['price_max'])}" \
            if s["priced_listings"] else "     -     "
        print(f"{fam:<22}{s['supply']:>7}{s['distinct_serials']:>8}"
              f"{money(s['price_median']):>11}{money(s['price_mean']):>11}"
              f"{rng:>22}")

    print()
    print("Top seller vocabulary (phrase -> listings):")
    for phrase, n in vocab.most_common(20):
        print(f"  {n:>4}  {phrase}")


def write_outputs(out_dir, stats, rows, vocab):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "report.json").write_text(json.dumps({
        "patterns": stats,
        "vocabulary": dict(vocab.most_common()),
        "n_listings": len(rows),
    }, indent=2))

    with (out / "patterns.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pattern", "supply", "distinct_serials", "priced_listings",
                    "price_min", "price_median", "price_mean", "price_max"])
        for fam, s in sorted(stats.items(),
                             key=lambda kv: kv[1]["supply"], reverse=True):
            w.writerow([fam, s["supply"], s["distinct_serials"],
                        s["priced_listings"], s["price_min"],
                        s["price_median"], s["price_mean"], s["price_max"]])

    with (out / "listings.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query", "item_id", "title", "price",
                                          "currency", "serial", "patterns",
                                          "url"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote report.json, patterns.csv, listings.csv -> {out}/")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Harvest eBay active fancy-serial listings for demand data.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-q", "--query", action="append", dest="queries",
                    help="Search term (repeatable). Overrides the default set.")
    ap.add_argument("--max-pages", type=int, default=2,
                    help="Pages (200 items each) per query. Default 2.")
    ap.add_argument("--category", default=None,
                    help="eBay category_ids filter (e.g. 3413 = US paper "
                         "money). Off by default.")
    ap.add_argument("--delay", type=float, default=0.5,
                    help="Seconds between paged calls. Default 0.5.")
    ap.add_argument("--out", default=None,
                    help="Directory for report.json + CSVs.")
    ap.add_argument("--sandbox", action="store_true",
                    help="Use eBay sandbox endpoints.")
    ap.add_argument("--client-id", default=None, help="eBay App ID.")
    ap.add_argument("--client-secret", default=None, help="eBay Cert ID.")
    ap.add_argument("--creds", default="~/.dbp_ebay.json",
                    help="JSON creds file. Default ~/.dbp_ebay.json.")
    ap.add_argument("--save-raw", default=None,
                    help="Write raw API item results to this JSON file.")
    ap.add_argument("--replay", default=None,
                    help="Read raw items from a JSON file instead of the API "
                         "(offline; no keys needed).")
    args = ap.parse_args(argv)

    queries = args.queries or DEFAULT_QUERIES

    if args.replay:
        raw = json.loads(Path(args.replay).read_text())
        items = raw["items"] if isinstance(raw, dict) else raw
        print(f"Replaying {len(items)} items from {args.replay}")
    else:
        cid, sec = resolve_creds(args)
        if not (cid and sec):
            ap.error(
                "No eBay keyset found. Provide --client-id/--client-secret, "
                "set EBAY_CLIENT_ID/EBAY_CLIENT_SECRET, or create "
                f"{args.creds}. See the module docstring for how to get a "
                "free keyset. (Or run --replay <file> offline.)")
        env = ENDPOINTS["sandbox" if args.sandbox else "production"]
        print("Requesting application token...")
        token = get_app_token(cid, sec, env["oauth"])
        print(f"Harvesting {len(queries)} queries "
              f"(up to {args.max_pages} pages each)...")
        items = harvest_live(queries, token, env["browse"], args.max_pages,
                             args.category, args.delay)
        if args.save_raw:
            Path(args.save_raw).write_text(
                json.dumps({"items": items}, indent=2))
            print(f"Saved {len(items)} raw items -> {args.save_raw}")

    stats, rows, vocab = build_report(items)
    print_summary(stats, rows, vocab, len(items))
    if args.out:
        write_outputs(args.out, stats, rows, vocab)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
