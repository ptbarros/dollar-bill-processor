"""Scanner Test logic against synthetic pages + a fake backend."""
import zipfile

import numpy as np

from dollar_detective.scanner import lab
from dollar_detective.scanner.backend import ScanError, ScanPage, ScannerDevice

DPI = 100
BG = (172, 167, 160)      # BGR grey backing
BILL = (200, 209, 205)


def page(long_edge_first=True, cut=False, pad_rows=40):
    """A raw page: grey strip with a bill, plus white padding after the paper."""
    bw, bh = (614, 261) if long_edge_first else (261, 614)
    bw, bh = int(bw * DPI / 100), int(bh * DPI / 100)
    W = 850 * DPI // 100
    H = bh + (0 if cut else 30) + 20
    img = np.full((H, W, 3), BG, np.uint8)
    x0 = (W - bw) // 2
    img[20:20 + bh, x0:x0 + bw] = BILL
    if cut:
        img = img[: 20 + bh - 5]          # scan ended before the paper did
    return np.vstack([img, np.full((pad_rows, W, 3), 255, np.uint8)])


def test_analyze_orientation_and_cut():
    a = lab.analyze_page(page(True), DPI)
    assert a["found"] and a["orientation"] == "long_edge_first"
    assert a["trailing_touch"] == 0.0
    assert abs(a["bill_in"][0] - 6.14) < 0.1
    b = lab.analyze_page(page(False), DPI)
    assert b["orientation"] == "short_edge_first"
    c = lab.analyze_page(page(True, cut=True), DPI)
    assert c["trailing_touch"] > 0.9


class Fake:
    label = "Fake"

    def __init__(self):
        self.calls = []

    def list_devices(self):
        return [ScannerDevice("d", "Fake R40")]

    def describe(self, device_id=None):
        return {"backend": "Fake", "name": "Fake R40", "feeder": {"can_scan_duplex": True}}

    def scan_pages(self, device_id=None, *, dpi, duplex, color, stop, scan_length_in=None,
                   transfer_format="jpeg"):
        self.calls.append(scan_length_in)
        self.formats = getattr(self, "formats", []) + [transfer_format]
        gap = 0.78 if transfer_format == "dib" else 1.1
        if self.empty:
            raise ScanError("No paper", cause="PE")
        for i in range(self.bills):
            yield ScanPage("A", page(), raw=page(), t=1.0 + i * gap)
            if duplex:
                yield ScanPage("B", page(), raw=page(), t=1.03 + i * gap)


def test_full_walkthrough_and_report(tmp_path):
    f = Fake()
    results = {}
    for step in lab.STEPS:
        f.empty = step.key == "empty"
        f.bills = step.default_bills
        r = lab.run_step(step, f, "d", results, bills=step.default_bills, dpi=DPI)
        if step.key == "empty":
            r.noise = "no"
        results[step.key] = r
        assert r.ok, (step.key, r.error)
    # 'short' used the measured window (bill end + margin), 'speed' the recommendation.
    w = lab.measured_window(results)
    assert w < 8.5 and f.calls[-3:] == [w, w, w]      # short, dib, speed
    assert f.formats[-3:] == ["jpeg", "dib", "dib"]   # speed uses the faster one
    s = lab.summarize(results)
    assert s["recommended"]["duplex"] is True
    assert s["recommended"]["feed"] == "long_edge_first"
    assert s["recommended"]["transfer_format"] == "dib"
    assert all(c["status"] == "pass" for c in s["checks"]), s["checks"]
    out = lab.save_report(tmp_path / "r.zip", results, s)
    names = zipfile.ZipFile(out).namelist()
    assert "report.txt" in names and "report.json" in names
    assert any(n.startswith("samples/") for n in names)


def test_duplex_page_count_mismatch_fails():
    f = Fake()
    f.empty, f.bills = False, 3
    results = {}
    r = lab.run_step(lab.STEP_BY_KEY["duplex"], f, "d", results, bills=4, dpi=DPI)
    results["duplex"] = r
    st = {c["check"]: c["status"] for c in lab.checks(results)}
    assert st["Two-sided scan"] == "fail"
    assert lab.recommend(results)["duplex"] is False
