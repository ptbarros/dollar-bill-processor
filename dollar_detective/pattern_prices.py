"""Group-editable pattern prices, keyed by catalog SKU.

Prices live in the per-pattern Lua `Price:` header as the shipped default. This
module adds an OPTIONAL override layer: a CSV the group can edit (in Excel/etc.)
and share, mapping a pattern's catalog SKU (e.g. "C-004") to a price string.
`engine.resolve_price()` consults these overrides first, then falls back to the
header -- so a missing CSV, or a pattern with no row, just shows its shipped
price. Nothing here is required for the app to run.

File: `<user_data_dir>/pattern_prices.csv`, columns:
    catalog, pattern, display_name, price, price_circ, price_unc
`catalog` is the key. `price` is the free-text display label (e.g. "$20-$100").
`price_circ` / `price_unc` are OPTIONAL numeric columns: `price_circ` is the
primary number used for the price column and sorting (circulated value), and
`price_unc` is an optional uncirculated value shown only where present. The
`pattern` / `display_name` columns are there so the file reads cleanly by hand.

Back-compat: an older 4-column file (catalog, pattern, display_name, price) still
loads -- the numeric columns are simply absent, and a sort/column number is then
derived from the low end of the free-text `price` string.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

FILENAME = "pattern_prices.csv"
HEADER = ["catalog", "pattern", "display_name", "price", "price_circ", "price_unc"]

_PRICE_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def parse_price_number(s: Optional[str]) -> Optional[float]:
    """The sort/column number for a price string: the FIRST (low-end) dollar
    amount found, so a range like "$20-$100" -> 20.0 and "$50" -> 50.0. Returns
    None when the string holds no number (e.g. "", "N/A", "Ask")."""
    if s is None:
        return None
    m = _PRICE_NUM_RE.search(str(s))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _fmt_num(x: Optional[float]) -> str:
    """Format a numeric price for the CSV: "" for None, no trailing ".0"."""
    if x is None:
        return ""
    return str(int(x)) if float(x).is_integer() else str(x)


def prices_path() -> Path:
    """Path to the user's editable prices CSV (may not exist yet)."""
    from dollar_detective.resource_path import user_data_dir
    return user_data_dir() / FILENAME


def load_overrides(path: Optional[Path] = None) -> dict:
    """Return {catalog_sku: {"display": str, "circ": float|None, "unc": float|None}}
    from the prices CSV, or {} if absent/unreadable.

    `display` is the free-text `price` column (the human label); `circ`/`unc` are
    the optional numeric columns. A row is kept when it has a catalog key AND at
    least one non-empty price field, so a fully blank row falls through to the
    shipped header price. Older 4-column files load fine (circ/unc come back
    None)."""
    p = Path(path) if path else prices_path()
    if not p.exists():
        return {}
    out = {}
    try:
        with p.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            # Be forgiving about header casing / stray spaces.
            fields = {(k or "").strip().lower(): k for k in (reader.fieldnames or [])}
            ckey = fields.get("catalog")
            pkey = fields.get("price")
            circ_key = fields.get("price_circ")
            unc_key = fields.get("price_unc")
            if not ckey or not (pkey or circ_key or unc_key):
                return {}
            for row in reader:
                sku = (row.get(ckey) or "").strip()
                display = (row.get(pkey) or "").strip() if pkey else ""
                circ = parse_price_number(row.get(circ_key)) if circ_key else None
                unc = parse_price_number(row.get(unc_key)) if unc_key else None
                if sku and (display or circ is not None or unc is not None):
                    out[sku] = {"display": display, "circ": circ, "unc": unc}
    except Exception:
        return {}
    return out


def _rows_from_engine(engine) -> list:
    """One (catalog, pattern, display_name, price, price_circ, price_unc) row per
    pattern that has a catalog SKU, using each pattern's current shipped/header
    price. `price_circ` is seeded from the low end of the header price so the
    numeric column starts populated; `price_unc` is left blank for the group to
    fill. Sorted by SKU so the file reads in catalog order."""
    rows = []
    for name, info in engine.lua_patterns.items():
        sku = (getattr(info, "catalog", "") or "").strip()
        if not sku:
            continue
        display = ""
        try:
            display = engine.get_pattern_info(name).get("display_name", name)
        except Exception:
            display = getattr(info, "display_name", "") or name
        header_price = getattr(info, "price", "") or ""
        circ = _fmt_num(parse_price_number(header_price))
        rows.append([sku, name, display, header_price, circ, ""])

    def _key(r):
        # Sort C-### / U-### numerically within a prefix, unknown formats last.
        sku = r[0]
        if "-" in sku and sku.rsplit("-", 1)[1].isdigit():
            prefix, num = sku.rsplit("-", 1)
            return (0, prefix, int(num))
        return (1, sku, 0)

    rows.sort(key=_key)
    return rows


def write_csv(rows: list, path: Path):
    """Write rows (each [catalog, pattern, display_name, price]) to a CSV."""
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)


def ensure_seeded(engine, path: Optional[Path] = None) -> Path:
    """Make sure every current pattern has a row in the prices CSV, returning the
    path.

    First time (no file yet) it writes the full baseline list. When the file
    already exists it TOPS IT UP: a row is appended (at the shipped header price)
    for every catalog SKU the engine has that isn't already in the file, so newly
    created user patterns and newly shipped core patterns become editable --
    while existing rows and the group's edits are left exactly as they are. New
    rows are appended, aligned to the file's current columns (so an older
    4-column file stays 4-column). An unreadable file is left untouched.
    """
    p = Path(path) if path else prices_path()
    engine_rows = _rows_from_engine(engine)
    if not p.exists():
        write_csv(engine_rows, p)
        return p

    # Read the existing file just enough to learn its columns and which SKUs it
    # already lists -- never rewrite the rows the user has, so edits are safe.
    try:
        with p.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
    except Exception:
        return p
    if not rows:
        write_csv(engine_rows, p)
        return p

    header = rows[0]
    lower = [(c or "").strip().lower() for c in header]
    ckey_idx = lower.index("catalog") if "catalog" in lower else 0
    existing = set()
    for r in rows[1:]:
        if r and ckey_idx < len(r):
            sku = (r[ckey_idx] or "").strip()
            if sku:
                existing.add(sku)

    # Map a canonical engine row onto whatever columns this file actually has,
    # so a top-up row lines up with the header (incl. legacy 4-column files).
    canon = {"catalog": 0, "pattern": 1, "display_name": 2,
             "price": 3, "price_circ": 4, "price_unc": 5}

    def to_file_cols(r):
        out = []
        for col in header:
            idx = canon.get((col or "").strip().lower())
            out.append(r[idx] if (idx is not None and idx < len(r)) else "")
        return out

    missing = [to_file_cols(r) for r in engine_rows if r[0] not in existing]
    if missing:
        with p.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(missing)
    return p


def export_to(engine, dest: Path, use_current: bool = True):
    """Write a shareable prices CSV to `dest`. With use_current (default) the
    prices are the live resolved values (overrides applied); otherwise the raw
    shipped header prices."""
    rows = []
    for r in _rows_from_engine(engine):
        sku, name, display, header_price, circ, unc = r
        price = header_price
        if use_current:
            try:
                price = engine.resolve_price(name)
            except Exception:
                price = header_price
            try:
                c, u = engine.resolve_price_number(name)
                circ = _fmt_num(c)
                unc = _fmt_num(u)
            except Exception:
                pass
        rows.append([sku, name, display, price, circ, unc])
    write_csv(rows, Path(dest))


def import_from(src: Path, path: Optional[Path] = None) -> int:
    """Load a shared prices CSV from `src` and make it the active prices file.
    Returns the number of priced rows imported. Raises on an unreadable file."""
    src = Path(src)
    overrides = load_overrides(src)
    if not overrides and src.exists():
        # File existed but had no usable catalog/price columns.
        raise ValueError("No 'catalog' and 'price' columns found in that CSV.")
    dest = Path(path) if path else prices_path()
    # Copy the raw file through so the human-readable columns are preserved.
    dest.write_text(src.read_text(encoding="utf-8-sig"), encoding="utf-8")
    return len(overrides)
