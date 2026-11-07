"""WinRT scanner backend: page ordering / side assignment / error mapping.

The WinRT calls only exist on Windows, so the scan coroutine is replaced with a
fake that writes files the way ScanFilesToFolderAsync does (SCN_<date>.jpg,
SCN_<date>_2.jpg, ...) and fires the progress callback.
"""

import threading
import time

import cv2
import numpy as np
import pytest

from dollar_detective.scanner import winrt_backend as wb
from dollar_detective.scanner.backend import ScanError, ScannerDevice


def _backend(fake_scan):
    b = object.__new__(wb.WinRtScannerBackend)   # skip the winrt import check
    b.list_devices = lambda: [ScannerDevice(id="dev1", name="Fake FF-640")]
    b._scan = fake_scan.__get__(b)
    return b


def _write(folder, name, value):
    img = np.full((40, 60, 3), value, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    (folder / name).write_bytes(buf.tobytes())


def _names(n):
    return ["SCN_20261107.jpg"] + [f"SCN_20261107_{i}.jpg" for i in range(2, n + 1)]


def test_duplex_pages_alternate_front_back_in_order():

    async def fake_scan(self, dev, folder, dpi, duplex, color, events, stop):
        events.put(("duplex", True))
        for i, name in enumerate(_names(24), 1):
            _write(folder, name, i * 10)       # brightness encodes page number
            time.sleep(0.005)                  # distinct creation times
            events.put(("page", i))

    pages = list(_backend(fake_scan).scan_pages("dev1", duplex=True))
    assert len(pages) == 24
    assert [p.side for p in pages] == ["A", "B"] * 12
    # Order preserved: _2 before _10 etc.
    vals = [int(p.image.mean()) for p in pages]
    assert vals == sorted(vals)


def test_simplex_all_fronts():
    async def fake_scan(self, dev, folder, dpi, duplex, color, events, stop):
        events.put(("duplex", False))
        for i, name in enumerate(_names(5), 1):
            _write(folder, name, 100)
            events.put(("page", i))

    pages = list(_backend(fake_scan).scan_pages("dev1", duplex=True))
    assert [p.side for p in pages] == ["A"] * 5


def test_missed_progress_callbacks_still_drained_at_end():
    async def fake_scan(self, dev, folder, dpi, duplex, color, events, stop):
        events.put(("duplex", True))
        for name in _names(4):
            _write(folder, name, 100)
            time.sleep(0.005)
        # no "page" events at all

    pages = list(_backend(fake_scan).scan_pages("dev1"))
    assert [p.side for p in pages] == ["A", "B", "A", "B"]


def test_empty_feeder_maps_to_PE():
    async def fake_scan(self, dev, folder, dpi, duplex, color, events, stop):
        e = OSError("paper empty")
        e.winerror = -2145320957       # 0x80210003 as a signed int
        raise e

    with pytest.raises(ScanError) as ei:
        list(_backend(fake_scan).scan_pages("dev1"))
    assert ei.value.cause == "PE"


def test_pages_before_error_are_kept():
    async def fake_scan(self, dev, folder, dpi, duplex, color, events, stop):
        events.put(("duplex", True))
        for i, name in enumerate(_names(2), 1):
            _write(folder, name, 100)
            time.sleep(0.005)
            events.put(("page", i))
        e = OSError("jam")
        e.winerror = 0x80210002
        raise e

    got = []
    with pytest.raises(ScanError) as ei:
        for p in _backend(fake_scan).scan_pages("dev1"):
            got.append(p.side)
    assert got == ["A", "B"]
    assert ei.value.cause == "JAM"


def test_stop_ends_scan():
    async def fake_scan(self, dev, folder, dpi, duplex, color, events, stop):
        import asyncio
        events.put(("duplex", True))
        i = 0
        for name in _names(200):
            if stop.is_set():
                return
            i += 1
            _write(folder, name, 100)
            events.put(("page", i))
            await asyncio.sleep(0.01)

    stop = threading.Event()
    n = 0
    for _ in _backend(fake_scan).scan_pages("dev1", stop=stop):
        n += 1
        if n == 6:
            stop.set()
            break
    assert n == 6


def test_unknown_device_falls_back_to_first():
    b = object.__new__(wb.WinRtScannerBackend)
    b.list_devices = lambda: [ScannerDevice(id="a", name="A"), ScannerDevice(id="b", name="B")]
    assert b._resolve_device("b") == "b"
    assert b._resolve_device("zzz") == "a"
    assert b._resolve_device(None) == "a"
    b.list_devices = lambda: []
    with pytest.raises(ScanError):
        b._resolve_device(None)
