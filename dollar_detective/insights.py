"""Build the Insights-report payload from the bill ledger.

Kept out of the GUI layer so it is testable without Qt and reusable by a preview
harness. ``build_payload`` turns a ``Ledger`` plus the live pattern engine into
the JSON object the bundled ``insights_template.html`` consumes.

What it adds over the raw ledger aggregates:
  * display names for every pattern (engine DisplayName / user label override),
    with the internal name kept alongside for tooltips;
  * several scan-date time windows (All time / 7d / 30d / 6mo / 12mo), each a
    small self-contained aggregate the report can switch between instantly, plus
    an optional custom range;
  * an estimated dollar value of the bills you kept, using the pattern prices CSV
    (``engine.resolve_price_number``); each kept bill is valued at the highest-
    priced pattern among its matches (no double counting), and value is also
    rolled up per pattern for a "where the value is" ranking;
  * a scanned -> fancy -> checked -> kept funnel (+ review backlog).

Everything is best-effort: a missing engine, missing prices, or an empty ledger
degrade gracefully rather than raise.
"""

from collections import Counter
from datetime import datetime, timedelta

# (key, label, days-back) — days None means all-time. Order here is the order
# the report's range dropdown shows them; "all" is the default active window.
WINDOW_SPECS = [
    ("all", "All time", None),
    ("week", "Last 7 days", 7),
    ("month", "Last 30 days", 30),
    ("m6", "Last 6 months", 182),
    ("year", "Last 12 months", 365),
]


def _price_map(engine):
    """{pattern_name: circulated_price_float} for every known pattern (0.0 when
    unpriced). Resolves CSV overrides then shipped Price: headers."""
    out = {}
    for name in getattr(engine, "lua_patterns", {}) or {}:
        try:
            circ, _unc = engine.resolve_price_number(name)
        except Exception:
            circ = None
        out[name] = float(circ) if circ else 0.0
    return out


def _display_fn(engine):
    """Memoised internal-name -> display-name lookup (falls back to the name)."""
    cache = {}

    def disp(name):
        if name in cache:
            return cache[name]
        label = name
        try:
            info = engine.get_pattern_info(name) if engine else None
            if info:
                label = info.get("display_name") or name
        except Exception:
            pass
        cache[name] = label
        return label

    return disp


def _filter_valid(patterns, valid):
    """Drop alias groups whose every name is a removed pattern (e.g. the pulled
    Green Guide), and trim each row's aliases to the names still present."""
    if not valid:
        return patterns
    kept = []
    for row in patterns:
        aliases = [a for a in (row.get("aliases") or [row.get("name")]) if a in valid]
        if not aliases:
            continue
        row = dict(row)
        row["aliases"] = aliases
        row["name"] = min(aliases, key=len)
        kept.append(row)
    return kept


def _value(kept_pattern_lists, price_map, disp):
    """Estimated value of kept bills. Each kept bill counts once, at the highest-
    priced pattern it matched; also rolled up per pattern for the ranking."""
    total = 0.0
    per = Counter()
    cnt = Counter()
    no_price = 0
    for pats in kept_pattern_lists:
        best, best_v = None, 0.0
        for p in pats:
            v = price_map.get(p, 0.0) or 0.0
            if v > best_v:
                best_v, best = v, p
        if best is None or best_v <= 0:
            no_price += 1
            continue
        total += best_v
        per[best] += best_v
        cnt[best] += 1
    top = [{"name": p, "display": disp(p), "value": round(per[p], 2), "count": cnt[p]}
           for p, _ in per.most_common(12)]
    return {
        "kept_value": round(total, 2),
        "top_by_value": top,
        "kept_priced": int(sum(cnt.values())),
        "kept_no_price": int(no_price),
    }


def _build_window(led, disp, price_map, enabled_names, valid, start, end, label):
    """One window of the report: filtered aggregates + display names + value +
    funnel, keyed by scan date (bills.first_seen in [start, end))."""
    rep = led.report_data(start=start, end=end)
    rep["patterns"] = _filter_valid(rep.get("patterns", []), valid)
    for row in rep["patterns"]:
        row["display"] = disp(row["name"])

    st = led.stats(start=start, end=end)
    kc = led.keep_coverage(enabled_names, start=start, end=end)
    # missing_patterns is [(name, count), ...]; append the display name.
    kc["missing_patterns"] = [[p, n, disp(p)] for p, n in kc.get("missing_patterns", [])]

    val = _value(led.kept_bill_patterns(start=start, end=end), price_map, disp)

    funnel = {
        "scanned": st.get("unique_bills", 0),
        "fancy": st.get("fancy", 0),
        "checked": st.get("checked", 0),
        "kept": st.get("kept", 0),
        "backlog": max(0, (st.get("fancy", 0) or 0) - (st.get("checked", 0) or 0)),
    }
    return {"label": label, "report": rep, "stats": st,
            "keep_coverage": kc, "value": val, "funnel": funnel}


def build_payload(led, engine, enabled_names, generated,
                  valid_names=None, custom=None):
    """Full Insights payload.

    led            : Ledger instance.
    engine         : pattern engine (for display names + prices); may be None.
    enabled_names  : set of currently-enabled pattern names (keep-coverage).
    generated      : timestamp string for the report header.
    valid_names    : optional set of pattern names still present in the engine;
                     rows whose every alias is absent are dropped.
    custom         : optional (start_iso, end_iso, label) extra window, made the
                     active one when present.
    """
    disp = _display_fn(engine)
    price_map = _price_map(engine) if engine else {}
    now = datetime.now()

    def cutoff(days):
        return (now - timedelta(days=days)).isoformat(timespec="seconds") if days else None

    windows, order = {}, []
    for key, label, days in WINDOW_SPECS:
        windows[key] = _build_window(
            led, disp, price_map, enabled_names, valid_names,
            cutoff(days), None, label)
        order.append(key)

    active = "all"
    if custom:
        c_start, c_end, c_label = custom
        windows["custom"] = _build_window(
            led, disp, price_map, enabled_names, valid_names,
            c_start, c_end, c_label or "Custom range")
        order.append("custom")
        active = "custom"

    return {
        "generated": generated,
        "windows": windows,
        "order": order,
        "active": active,
        "prices_available": any(v > 0 for v in price_map.values()),
    }
