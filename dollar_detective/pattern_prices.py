"""Group-editable pattern prices, keyed by catalog SKU.

Prices live in the per-pattern Lua `Price:` header as the shipped default. This
module adds an OPTIONAL override layer: a CSV the group can edit (in Excel/etc.)
and share, mapping a pattern's catalog SKU (e.g. "C-004") to a price string.
`engine.resolve_price()` consults these overrides first, then falls back to the
header -- so a missing CSV, or a pattern with no row, just shows its shipped
price. Nothing here is required for the app to run.

File: `<user_data_dir>/pattern_prices.csv`, columns:
    catalog, pattern, display_name, price
Only `catalog` (the key) and `price` are read back; the other two are there so
the file is readable when edited by hand.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

FILENAME = "pattern_prices.csv"
HEADER = ["catalog", "pattern", "display_name", "price"]


def prices_path() -> Path:
    """Path to the user's editable prices CSV (may not exist yet)."""
    from dollar_detective.resource_path import user_data_dir
    return user_data_dir() / FILENAME


def load_overrides(path: Optional[Path] = None) -> dict:
    """Return {catalog_sku: price} from the prices CSV, or {} if absent/unreadable.

    Only rows with a non-empty catalog key AND a non-empty price are kept, so a
    blank price row falls through to the shipped header price."""
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
            if not ckey or not pkey:
                return {}
            for row in reader:
                sku = (row.get(ckey) or "").strip()
                price = (row.get(pkey) or "").strip()
                if sku and price:
                    out[sku] = price
    except Exception:
        return {}
    return out


def _rows_from_engine(engine) -> list:
    """One (catalog, pattern, display_name, price) row per pattern that has a
    catalog SKU, using each pattern's current shipped/header price. Sorted by
    SKU so the file reads in catalog order."""
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
        rows.append([sku, name, display, getattr(info, "price", "") or ""])

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
    """Create the prices CSV seeded from current prices if it doesn't exist yet,
    so the first edit starts from a full, baseline list. Returns the path."""
    p = Path(path) if path else prices_path()
    if not p.exists():
        write_csv(_rows_from_engine(engine), p)
    return p


def export_to(engine, dest: Path, use_current: bool = True):
    """Write a shareable prices CSV to `dest`. With use_current (default) the
    prices are the live resolved values (overrides applied); otherwise the raw
    shipped header prices."""
    rows = []
    for r in _rows_from_engine(engine):
        sku, name, display, header_price = r
        price = header_price
        if use_current:
            try:
                price = engine.resolve_price(name)
            except Exception:
                price = header_price
        rows.append([sku, name, display, price])
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
