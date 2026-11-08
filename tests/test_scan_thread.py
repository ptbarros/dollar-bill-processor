"""ScanThread: stays armed across feeder loads, numbers monotonically, keeps
going past an empty feeder, writes front-only leftovers, records progress on
failure."""
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
from gui.scan_thread import ScanThread                     # noqa: E402
from dollar_detective.scanner.backend import ScanError, ScanPage   # noqa: E402

IMG = np.full((20, 30, 3), 128, np.uint8)


class FakeBackend:
    can_detect_paper = False

    def __init__(self, passes, stop_after):
        self.passes = list(passes)    # each: list of sides, or an Exception
        self.stop_after = stop_after
        self.calls = 0

    def scan_pages(self, device_id=None, *, dpi, duplex, color, stop):
        self.calls += 1
        item = self.passes.pop(0) if self.passes else []
        if self.calls >= self.stop_after:
            stop.set()
        if isinstance(item, Exception):
            raise item
        for side in item:
            yield ScanPage(side=side, image=IMG)

    def wait_for_paper(self, device_id, stop):
        return not stop.is_set()


def _run(backend, start=None):
    out = Path(tempfile.mkdtemp())
    t = ScanThread(backend, out, start_number=start)
    res = {}
    t.finished_ok.connect(lambda n: res.setdefault("ok", n))
    t.failed.connect(lambda m: res.setdefault("fail", m))
    t.run()                       # synchronously
    return t, out, res


def names(out):
    return sorted(p.name for p in out.iterdir() if not p.name.startswith("."))


def test_armed_across_loads_and_empty_feeder():
    b = FakeBackend([["A", "B", "A", "B"], ScanError("empty", cause="PE"),
                     ["A", "A", "B", "B"]], stop_after=3)
    t, out, res = _run(b, start=5)
    assert b.calls == 3
    assert names(out) == ["FastFoto_00005.jpg", "FastFoto_00005_b.jpg",
                          "FastFoto_00006.jpg", "FastFoto_00006_b.jpg",
                          "FastFoto_00007.jpg", "FastFoto_00007_b.jpg",
                          "FastFoto_00008.jpg", "FastFoto_00008_b.jpg"]
    assert t.final_number == 8 and res == {"ok": 8}


def test_leftover_front_written_and_numbering_kept_on_failure():
    b = FakeBackend([["A", "B", "A"], ScanError("boom", cause="JAM")], stop_after=99)
    t, out, res = _run(b, start=1)
    assert "fail" in res
    assert names(out) == ["FastFoto_00001.jpg", "FastFoto_00001_b.jpg",
                          "FastFoto_00002.jpg"]
    assert t.final_number == 2
