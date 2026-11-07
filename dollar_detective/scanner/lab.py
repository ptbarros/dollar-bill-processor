"""Scanner Test: find out what a feed scanner can do, step by step.

Drives the app's own scanner backend (so a pass here means the app's Scan
button will work), measures every page, and turns the measurements into a
plain checklist + recommended settings + a report the user can send back.

No GUI here -- gui/scanner_lab_dialog.py walks the user through STEPS and calls
run_step(); summarize() and save_report() work on the collected results.
"""

from __future__ import annotations

import io
import json
import platform
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from . import billcrop
from .backend import ScanError
from .padding import trim_padding

BILL_LONG_IN = 6.14      # US note, 156 mm
BILL_SHORT_IN = 2.61     # 66.3 mm
WINDOW_MARGIN_IN = 0.75  # headroom added to the measured bill length
SAMPLE_MAX_W = 1200      # downscale sample images in the report


@dataclass
class Step:
    key: str
    title: str
    instructions: str
    scan: Optional[dict] = None        # scan options; None = no scanning
    default_bills: int = 0             # 0 = no paper for this step
    ask_noise: bool = False


STEPS: List[Step] = [
    Step("discover", "Find the scanner",
         "Plug the scanner in, switch it on, and CLOSE the scanner's own "
         "software (check the system tray too) so it isn't holding the "
         "scanner.\n\nNo paper needed. Click Run to look for scanners and read "
         "what they report they can do."),
    Step("empty", "Empty feeder",
         "Take ALL paper out of the feeder, then click Run.\n\nThis checks that "
         "an empty feeder is noticed quickly and quietly -- the app relies on "
         "that to wait for your next stack of bills between loads.",
         scan={"duplex": True, "scan_length_in": None}, ask_noise=True),
    Step("front", "Front only",
         "Load 3 bills exactly the way you normally feed them, then click Run.\n\n"
         "Scans one side only, using the scanner's full scan length, to see "
         "where the bills land and which way they feed.",
         scan={"duplex": False, "scan_length_in": None, "transfer_format": "jpeg"},
         default_bills=3),
    Step("duplex", "Both sides",
         "Load 3 bills the same way, then click Run.\n\nScans front and back in "
         "one pass (duplex), the way the app scans a strap.",
         scan={"duplex": True, "scan_length_in": None, "transfer_format": "jpeg"},
         default_bills=3),
    Step("short", "Shorter scan window",
         "Load 3 bills the same way, then click Run.\n\nScans only as far as the "
         "bills actually reach (measured in the earlier steps) -- usually faster. "
         "Checks the bills still come out whole.",
         scan={"duplex": True, "scan_length_in": "measured", "transfer_format": "jpeg"},
         default_bills=3),
    Step("dib", "Uncompressed transfer",
         "Load 5 bills, then click Run.\n\nSends the pages uncompressed. Some "
         "scanners compress more slowly than they feed, so this can be much "
         "faster (the FastFoto FF-640 went from 1.12 to 0.78 s per bill).",
         scan={"duplex": True, "scan_length_in": "measured", "transfer_format": "dib"},
         default_bills=5),
    Step("speed", "Speed run",
         "Load 10 bills, then click Run.\n\nScans with the best settings found "
         "so far, to measure real bills-per-minute.",
         scan={"duplex": True, "scan_length_in": "recommended",
               "transfer_format": "recommended"}, default_bills=10),
]
STEP_BY_KEY = {s.key: s for s in STEPS}


# --------------------------------------------------------------------------- #
# Page analysis
# --------------------------------------------------------------------------- #
def analyze_page(raw: np.ndarray, dpi: int) -> dict:
    """Measure one raw page: where the bill is, which way it fed, whether it
    runs into the end of the scan (cut off), and the background colour."""
    h0, w0 = raw.shape[:2]
    out = {"w": int(w0), "h": int(h0)}
    try:
        trimmed = billcrop._trim_white_padding(trim_padding(raw))
        out["white_pad_rows"] = int(h0 - trimmed.shape[0])
        bg = billcrop._background_colour(trimmed)
        out["background_rgb"] = [int(bg[2]), int(bg[1]), int(bg[0])]
        mask, _c = billcrop._item_mask(trimmed)
    except Exception as e:
        out["found"] = False
        out["error"] = repr(e)
        return out
    if mask is None:
        out["found"] = False
        return out
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    th, tw = mask.shape
    cols = mask[:, x0:x1 + 1]
    bw, bh = (x1 - x0 + 1) / dpi, (y1 - y0 + 1) / dpi
    out.update({
        "found": True,
        "bill_in": [round(bw, 2), round(bh, 2)],
        # Paper travels down the image. Bill wider than tall = long edge led.
        "orientation": "long_edge_first" if bw >= bh else "short_edge_first",
        "bill_top_in": round(y0 / dpi, 2),
        "bill_end_in": round((y1 + 1) / dpi, 2),
        "side_margins_in": [round(x0 / dpi, 2), round((tw - 1 - x1) / dpi, 2)],
        # Fraction of the bill's width touching the first/last real row: high
        # = the page started/ended on the bill, i.e. that edge was cut off.
        "leading_touch": round(float(cols[0].mean()), 2),
        "trailing_touch": round(float(cols[-1].mean()), 2) if y1 >= th - 1 else 0.0,
    })
    return out


# --------------------------------------------------------------------------- #
# Running a step
# --------------------------------------------------------------------------- #
@dataclass
class StepResult:
    key: str
    ok: bool = False
    skipped: bool = False
    error: str = ""
    cause: str = ""
    bills_loaded: int = 0
    options: dict = field(default_factory=dict)
    seconds: float = 0.0
    pages: List[dict] = field(default_factory=list)   # side, t, analysis
    info: dict = field(default_factory=dict)          # discover output
    noise: Optional[str] = None                       # user's answer (empty step)
    samples: Dict[str, bytes] = field(default_factory=dict)   # name -> JPEG


def _jpeg(img, max_w=SAMPLE_MAX_W) -> bytes:
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


def bill_whole(a: dict, tol_in: float = 0.1) -> bool:
    """Did the whole bill make it into the scan? Judged by its length along the
    feed (2.61" long-edge-first, 6.14" short-edge-first), NOT by whether it
    touches the end of the page: many drivers stop exactly at the paper's end,
    so a complete bill always touches the last row."""
    if not a.get("found"):
        return False
    expected = BILL_SHORT_IN if a.get("orientation") == "long_edge_first" else BILL_LONG_IN
    return a["bill_in"][1] >= expected - tol_in


def resolve_scan_length(option, results: Dict[str, StepResult]) -> Optional[float]:
    """Turn 'measured' / 'recommended' into inches (None = full length)."""
    if option == "measured":
        return measured_window(results)
    if option == "recommended":
        return recommend(results).get("scan_length_in")
    return option


def measured_window(results: Dict[str, StepResult]) -> Optional[float]:
    ends = [p["analysis"]["bill_end_in"]
            for k in ("front", "duplex") if k in results
            for p in results[k].pages if p["analysis"].get("found")]
    if not ends:
        return None
    return round(max(ends) + WINDOW_MARGIN_IN, 2)


def run_step(step: Step, backend, device_id, results: Dict[str, StepResult], *,
             bills: int = 0, dpi: int = 300, stop=None,
             progress: Callable[[str], None] = lambda s: None) -> StepResult:
    res = StepResult(key=step.key, bills_loaded=bills)
    t0 = time.monotonic()
    if step.scan is None:
        try:
            res.info = backend.describe(device_id)
            res.ok = bool(backend.list_devices())
            if not res.ok:
                res.error = "No scanner found."
        except Exception as e:
            res.error = str(e)
        res.seconds = round(time.monotonic() - t0, 2)
        return res

    opts = dict(step.scan)
    opts["scan_length_in"] = resolve_scan_length(opts.get("scan_length_in"), results)
    if opts.get("transfer_format") == "recommended":
        opts["transfer_format"] = recommend(results).get("transfer_format", "jpeg")
    res.options = dict(opts, dpi=dpi)
    progress("Scanning…")
    fronts = backs = 0
    try:
        for page in backend.scan_pages(device_id, dpi=dpi, color=True, stop=stop, **opts):
            raw = page.raw if page.raw is not None else page.image
            a = analyze_page(raw, dpi)
            res.pages.append({"side": page.side, "t": round(page.t, 2), "analysis": a})
            if page.side == "B":
                backs += 1
            else:
                fronts += 1
            n = len(res.pages)
            if n <= 2:   # keep the first front/back as samples
                res.samples[f"{step.key}_{n}_{page.side}_raw.jpg"] = _jpeg(raw)
                res.samples[f"{step.key}_{n}_{page.side}_crop.jpg"] = _jpeg(page.image)
            progress(f"{n} page{'s' if n != 1 else ''} ({fronts} front, {backs} back)…")
        res.ok = True
    except ScanError as e:
        res.error, res.cause = str(e), getattr(e, "cause", "")
        # An empty feeder reporting "paper empty" is a pass for that step.
        res.ok = step.key == "empty" and res.cause in ("PE", "PJ") and not res.pages
    except Exception as e:
        res.error = repr(e)
    res.seconds = round(time.monotonic() - t0, 2)
    return res


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #
def _rate(res: StepResult) -> Optional[float]:
    """Seconds per bill once feeding, from page arrival times."""
    ts = sorted(p["t"] for p in res.pages)
    if len(ts) < 2:
        return None
    sides = 2 if any(p["side"] == "B" for p in res.pages) else 1
    bills = len(ts) / sides
    if bills < 2:
        return None
    return round((ts[-1] - ts[0]) / (bills - 1), 2)


def _streams(res: StepResult) -> Optional[bool]:
    """True if pages arrive while feeding (not all at the end)."""
    ts = sorted(p["t"] for p in res.pages)
    if len(ts) < 3:
        return None
    # All-at-the-end delivery packs every page into a fraction of a second.
    return (ts[-1] - ts[0]) / (len(ts) - 1) > 0.15


def checks(results: Dict[str, StepResult]) -> List[dict]:
    """Plain-language pass/fail lines. status: pass | warn | fail | skip."""
    out = []

    def add(name, status, detail):
        out.append({"check": name, "status": status, "detail": detail})

    d = results.get("discover")
    if d:
        if d.ok:
            name = d.info.get("name") or d.info.get("backend", "")
            add("Scanner found", "pass", name)
            feeder = d.info.get("feeder", {})
            if feeder:
                add("Reports duplex", "pass" if feeder.get("can_scan_duplex") is True else "warn",
                    str(feeder.get("can_scan_duplex")))
        else:
            add("Scanner found", "fail", d.error or "not found")

    e = results.get("empty")
    if e and not e.skipped:
        quick = e.seconds < 3 and not e.pages
        quiet = e.noise != "yes"
        add("Empty feeder noticed quickly", "pass" if (e.ok and quick) else "fail",
            f"{e.seconds:.2f}s, {len(e.pages)} pages" + (f", {e.cause or e.error}" if e.error else ""))
        if e.noise:
            add("Empty feeder is quiet", "pass" if quiet else "warn",
                "no roller movement" if quiet else "rollers moved -- waiting between loads may be noisy")

    for key, label, per_bill in (("front", "Front-only scan", 1), ("duplex", "Two-sided scan", 2)):
        r = results.get(key)
        if not r or r.skipped:
            continue
        want = r.bills_loaded * per_bill
        if not r.ok:
            add(label, "fail", r.error or "failed")
            continue
        got = len(r.pages)
        add(label, "pass" if got == want else "fail",
            f"{got} pages for {r.bills_loaded} bills (expected {want})")
        found = [p["analysis"] for p in r.pages]
        bad = sum(1 for a in found if not a.get("found"))
        add(f"{label}: bill found on background", "pass" if not bad else "warn",
            "all pages" if not bad else f"{bad} of {len(found)} pages -- background may be too close to the bill's colour")
        seen = [a for a in found if a.get("found")]
        if seen:
            short = [a for a in seen if not bill_whole(a)]
            touch = max(a["trailing_touch"] for a in seen)
            if short:
                worst = min(a["bill_in"][1] for a in short)
                add(f"{label}: whole bill scanned", "fail",
                    f"{len(short)} of {len(seen)} bills came out short ({worst:.2f} in "
                    "along the feed) -- the scan window is cutting them off")
            else:
                add(f"{label}: whole bill scanned", "pass",
                    "full length" + ("; the scanner stops right at the paper's end, so a "
                                     "crooked bill can lose a corner sliver (normal for "
                                     "this driver)" if touch >= 0.6 else ""))
        s = _streams(r)
        if s is not None:
            add(f"{label}: pages arrive as they feed", "pass" if s else "warn",
                "streams page by page" if s else "pages seem to arrive all at once at the end")
        rate = _rate(r)
        if rate:
            add(f"{label}: speed", "pass", f"{rate:.2f} s per bill ({60 / rate:.0f} bills/min)")

    sh = results.get("short")
    du = results.get("duplex")
    if sh and not sh.skipped and sh.ok and du and du.ok:
        whole = all(bill_whole(p["analysis"]) for p in sh.pages)
        r_full, r_short = _rate(du), _rate(sh)
        gain = (f"{r_full:.2f} → {r_short:.2f} s/bill" if (r_full and r_short) else "")
        add("Shorter window keeps bills whole", "pass" if whole else "fail",
            f"window {sh.options.get('scan_length_in')} in. {gain}")

    dib = results.get("dib")
    base = results.get("short") or results.get("duplex")
    if dib and not dib.skipped:
        if dib.ok:
            r_dib, r_base = _rate(dib), (_rate(base) if base and base.ok else None)
            faster = bool(r_dib and r_base and r_dib < 0.9 * r_base)
            add("Uncompressed transfer", "pass" if faster else "warn",
                (f"{r_base:.2f} → {r_dib:.2f} s/bill" if (r_dib and r_base) else "measured")
                + (" -- faster, will use it" if faster else " -- no real gain, keeping JPEG"))
        else:
            add("Uncompressed transfer", "warn", f"not supported ({dib.error})")

    sp = results.get("speed")
    if sp and not sp.skipped:
        if sp.ok:
            sides = 2 if sp.options.get("duplex") else 1
            got, want = len(sp.pages), sp.bills_loaded * sides
            rate = _rate(sp)
            add("Speed run", "pass" if got == want else "fail",
                f"{got}/{want} pages" + (f", {rate:.2f} s/bill ({60 / rate:.0f} bills/min)" if rate else ""))
        else:
            add("Speed run", "fail", sp.error)
    return out


def orientation(results: Dict[str, StepResult]) -> Optional[str]:
    votes = [p["analysis"]["orientation"]
             for k in ("front", "duplex", "short") if k in results
             for p in results[k].pages if p["analysis"].get("found")]
    return max(set(votes), key=votes.count) if votes else None


def recommend(results: Dict[str, StepResult]) -> dict:
    """Settings the app should use for this scanner, from what passed."""
    rec: dict = {}
    du = results.get("duplex")
    rec["duplex"] = bool(du and du.ok and len(du.pages) == 2 * du.bills_loaded
                         and du.bills_loaded > 0)
    rec["scan_length_in"] = None
    sh = results.get("short")
    if sh and sh.ok and sh.pages and all(bill_whole(p["analysis"]) for p in sh.pages):
        rec["scan_length_in"] = sh.options.get("scan_length_in")
    rec["feed"] = orientation(results)
    rec["transfer_format"] = "jpeg"
    dib, base = results.get("dib"), results.get("short") or results.get("duplex")
    if dib and dib.ok and base and base.ok:
        r_dib, r_base = _rate(dib), _rate(base)
        if r_dib and r_base and r_dib < 0.9 * r_base and len(dib.pages) == 2 * dib.bills_loaded:
            rec["transfer_format"] = "dib"
    return rec


def summarize(results: Dict[str, StepResult]) -> dict:
    return {"checks": checks(results), "recommended": recommend(results)}


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def report_text(results: Dict[str, StepResult], summary: dict) -> str:
    sym = {"pass": "OK  ", "warn": "WARN", "fail": "FAIL", "skip": "skip"}
    lines = ["Dollar Detective -- Scanner Test report", ""]
    d = results.get("discover")
    if d and d.info:
        lines.append(f"Scanner: {d.info.get('name') or '?'}  ({d.info.get('backend', '')})")
    lines.append(f"Computer: {platform.platform()}")
    lines.append("")
    for c in summary["checks"]:
        lines.append(f"[{sym.get(c['status'], c['status'])}] {c['check']}: {c['detail']}")
    lines += ["", "Recommended settings:"]
    for k, v in summary["recommended"].items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines) + "\n"


def save_report(path, results: Dict[str, StepResult], summary: dict,
                extra: Optional[dict] = None) -> Path:
    """Write a zip: report.txt (readable), report.json (everything), samples."""
    path = Path(path)
    data = {
        "app": extra or {},
        "platform": platform.platform(),
        "summary": summary,
        "steps": {k: {kk: vv for kk, vv in r.__dict__.items() if kk != "samples"}
                  for k, r in results.items()},
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.txt", report_text(results, summary))
        z.writestr("report.json", json.dumps(data, indent=2, default=str))
        for r in results.values():
            for name, blob in r.samples.items():
                if blob:
                    z.writestr(f"samples/{name}", blob)
    return path
