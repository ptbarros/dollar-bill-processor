"""TWAIN backend: helper protocol handling, source matching, Auto fallback.
A fake helper (Python script) stands in for dd_twain_helper.exe."""
import os
import stat
import sys
import textwrap
import threading
import time

import pytest

from dollar_detective.scanner import twain_backend as tb
from dollar_detective.scanner.backend import ScanError, ScanPage, ScannerDevice

FAKE = textwrap.dedent(r'''
    #!{py}
    import json, sys, os, time
    import numpy as np, cv2
    args = sys.argv[1:]
    def emit(**k): print(json.dumps(k), flush=True)
    mode = os.environ.get("FAKE_MODE", "ok")
    if args[0] == "list":
        emit(event="source", name="EPSON FF-640", manufacturer="EPSON", family="FF", protocol="2.4")
        emit(event="end"); sys.exit(0)
    if args[0] == "caps":
        emit(event="caps", source=args[2], caps={{"0x1150": {{"one": True}}}}); emit(event="end"); sys.exit(0)
    out = args[args.index("--out") + 1]
    if mode == "fail":
        emit(event="error", rc=1, cc=4, msg="MSG_OPENDS failed"); emit(event="end"); sys.exit(6)
    if mode == "empty":
        emit(event="config", source="EPSON FF-640", border_detect=True, feeder_sensor=True)
        emit(event="empty", t=0.5); emit(event="end"); sys.exit(0)
    if mode == "jam":
        emit(event="config", source="EPSON FF-640", border_detect=True, feeder_sensor=True)
        emit(event="error", rc=0, cc=0, recoverable=True, msg="the scanner stopped the batch (jam, double feed or cover open?)")
        emit(event="end"); sys.exit(0)
    emit(event="config", source="EPSON FF-640", border_detect=True, feeder_sensor=True)
    n = 0
    for batch in range(2):
        emit(event="waiting", t=0)
        for i in range(4):
            n += 1
            p = os.path.join(out, "page_%05d.jpg" % n)
            ok, buf = cv2.imencode(".jpg", np.full((30, 70, 3), 100 + n, np.uint8))
            open(p, "wb").write(buf.tobytes())
            emit(event="page", n=n, side="A" if i % 2 == 0 else "B", file=p, t=n * 0.6)
        emit(event="batch_end", pages=4, t=n)
    while not os.path.exists(os.path.join(out, "STOP")):
        time.sleep(0.05)
    emit(event="end")
''')


@pytest.fixture
def helper(tmp_path, monkeypatch):
    exe = tmp_path / "dd_twain_helper.exe"
    exe.write_text(FAKE.format(py=sys.executable).lstrip())
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("DD_TWAIN_HELPER", str(exe))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    return exe


def test_match_source():
    srcs = [{"name": "EPSON FF-640"}, {"name": "CANON R40"}, {"name": "WIA-EPSON FF-640"}]
    assert tb.match_source("Epson FF-640", srcs)["name"] == "EPSON FF-640"
    assert tb.match_source("CANON R40 USB #3", srcs)["name"] == "CANON R40"
    assert tb.match_source("Fujitsu fi-800R", srcs) is None


def test_list_and_scan_across_stacks_until_stop(helper):
    b = tb.TwainBackend()
    assert [d.name for d in b.list_devices()] == ["EPSON FF-640"]
    stop = threading.Event()
    sides = []
    for page in b.scan_pages("twain:EPSON FF-640", stop=stop):
        sides.append(page.side)
        assert page.image.shape == (30, 70, 3)     # driver crops -> passed through
        if len(sides) == 8:
            stop.set()                              # helper waits for STOP
    assert sides == ["A", "B"] * 4


def test_helper_error_raises(helper, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "fail")
    with pytest.raises(ScanError) as ei:
        list(tb.TwainBackend().scan_pages("twain:EPSON FF-640", stop=threading.Event()))
    assert ei.value.cause == "TWAIN_START"      # couldn't open: may fall back


def test_epson_settings_restored(helper, tmp_path):
    root = tmp_path / "appdata" / "Epson" / "Epson Scan 2" / "Settings" / "ES012B"
    root.mkdir(parents=True)
    (root / "UserSettings.SF2").write_bytes(b"original")
    with tb._EpsonSettingsGuard():
        (root / "UserSettings.SF2").write_bytes(b"driver rewrote it")
        (root / "New.SF2").write_bytes(b"new")
    assert (root / "UserSettings.SF2").read_bytes() == b"original"
    assert not (root / "New.SF2").exists()


class FakeWia:
    label = "wia"

    def list_devices(self):
        return [ScannerDevice("wia-id", "Epson FF-640")]

    def scan_pages(self, device_id=None, **k):
        yield ScanPage("A", None)


def _windows(helper, wia=True):
    w = object.__new__(tb.WindowsScannerBackend)
    w.wia = FakeWia() if wia else None
    w.twain = tb.TwainBackend()
    w._twain_failed = False
    w.last_driver = None
    return w


def test_auto_prefers_twain_and_wia_forces_wia(helper):
    w = _windows(helper)
    b, dev = w.choose("wia-id", "auto")
    assert b is w.twain and dev == "twain:EPSON FF-640"
    b, dev = w.choose("wia-id", "wia")
    assert b is w.wia and dev == "wia-id"


def test_auto_falls_back_to_wia_only_when_twain_cant_start(helper, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "fail")          # "MSG_OPENDS failed"
    w = _windows(helper)
    pages = list(w.scan_pages("wia-id", stop=threading.Event(), driver="auto"))
    assert len(pages) == 1 and w.last_driver == "wia"
    # not sticky: the next scan tries TWAIN again
    monkeypatch.setenv("FAKE_MODE", "ok")
    b, _ = w.choose("wia-id", "auto")
    assert b is w.twain
    # forced TWAIN never falls back
    monkeypatch.setenv("FAKE_MODE", "fail")
    with pytest.raises(ScanError):
        list(w.scan_pages("wia-id", stop=threading.Event(), driver="twain"))


def test_empty_feeder_does_not_fall_back_to_wia(helper, monkeypatch):
    """The v1.15.0 bug: TWAIN's 'no paper' sent the Scanner Test to WIA."""
    monkeypatch.setenv("FAKE_MODE", "empty")
    w = _windows(helper)
    with pytest.raises(ScanError) as ei:
        list(w.scan_pages("wia-id", stop=threading.Event(), driver="auto", once=True))
    assert ei.value.cause == "PE" and w.last_driver == "twain"


def test_jam_does_not_fall_back_to_wia(helper, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "jam")
    w = _windows(helper)
    with pytest.raises(ScanError) as ei:
        list(w.scan_pages("wia-id", stop=threading.Event(), driver="auto"))
    assert ei.value.cause == "TWAIN" and w.last_driver == "twain"


def test_describe_does_not_open_wia_when_twain_is_used(helper):
    """v1.15.1: reading WIA capabilities alone crashed the Windows scanner
    service on Paul's Epson; with TWAIN in use, never open WIA."""
    w = _windows(helper)
    opened = []
    w.wia.describe = lambda *a, **k: opened.append(1) or {"name": "x"}
    info = w.describe("wia-id")
    assert info["driver"] == "twain" and not opened
    assert info["wia"]["skipped"] and info["twain"]["caps"]
    w.describe("wia-id", driver="wia")
    assert opened                                  # explicitly asked for WIA
