"""WinRT scanner backend: page hand-off logic (ordering, sides, completeness,
stop, errors), run against a fake async op + a thread that writes files the
way ScanFilesToFolderAsync does (SCN_<date>.jpg, SCN_<date>_2.jpg, ...).
The real WinRT calls only exist on Windows."""

import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from dollar_detective.scanner import winrt_backend as wb
from dollar_detective.scanner.backend import ScanError, ScannerDevice

wb.STEADY_S = 0.15
wb.POLL_S = 0.02
wb.READ_RETRY_S = 0.5


class FakeOp:
    def __init__(self):
        self._status = 0
        self.error_code = 0
        self.cancelled = False
        self.not_started_reads = 0   # first N status reads throw NOT_STARTED

    @property
    def status(self):
        if self.not_started_reads > 0:
            self.not_started_reads -= 1
            e = OSError("not started")
            e.winerror = -2147483623   # 0x80000019
            raise e
        return self._status

    def cancel(self):
        self.cancelled = True
        if self._status == 0:
            self._status = 2


def _jpeg(value):
    ok, buf = cv2.imencode(".jpg", np.full((40, 60, 3), value, np.uint8))
    return buf.tobytes()


def _names(n):
    return ["SCN_20261107.jpg"] + [f"SCN_20261107_{i}.jpg" for i in range(2, n + 1)]


def _writer(folder, op, n, *, gap=0.01, end_status=1, error_code=0,
            pause_after=None, partial=None, stop_writing=None):
    def run():
        for i, name in enumerate(_names(n), 1):
            if op.cancelled or (stop_writing and stop_writing.is_set()):
                break
            data = _jpeg(min(250, i * 10))
            if partial == i:
                (folder / name).write_bytes(data[: len(data) // 3])
                time.sleep(0.3)      # being written slowly
            (folder / name).write_bytes(data)
            time.sleep(gap)
            if pause_after == i:
                time.sleep(0.4)
        if not op.cancelled:
            op.error_code = error_code
            op._status = end_status
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def _pump(op, folder, duplex=True, stop=None):
    b = object.__new__(wb.WinRtScannerBackend)
    return b._pump(op, folder, duplex, stop, time.monotonic())


def _folder():
    return Path(tempfile.mkdtemp(prefix="dd_test_scan_"))


def test_duplex_order_and_sides():
    op, folder = FakeOp(), _folder()
    _writer(folder, op, 24)
    pages = list(_pump(op, folder))
    assert [p.side for p in pages] == ["A", "B"] * 12
    vals = [int(p.image.mean()) for p in pages]
    assert vals == sorted(vals)            # _2 before _10 etc.
    assert not folder.exists()             # temp dir cleaned


def test_simplex_all_fronts():
    op, folder = FakeOp(), _folder()
    _writer(folder, op, 5)
    assert [p.side for p in _pump(op, folder, duplex=False)] == ["A"] * 5


def test_not_started_status_is_tolerated():
    op, folder = FakeOp(), _folder()
    op.not_started_reads = 5
    _writer(folder, op, 4)
    assert len(list(_pump(op, folder))) == 4


def test_last_page_of_a_pause_is_handed_on_before_op_ends():
    """The back of the last bill before a reload must not wait for the op."""
    op, folder = FakeOp(), _folder()
    stop_writing = threading.Event()
    got = []

    def consume():
        for p in _pump(op, folder):
            got.append(p.side)
    _writer(folder, op, 4, stop_writing=stop_writing, end_status=0)
    c = threading.Thread(target=consume, daemon=True)
    c.start()
    time.sleep(0.6)                        # op still "running", writer done
    assert got == ["A", "B", "A", "B"]
    op._status = 1
    c.join(2)


def test_partially_written_file_waits_not_dropped():
    op, folder = FakeOp(), _folder()
    _writer(folder, op, 6, partial=4)
    pages = list(_pump(op, folder))
    assert len(pages) == 6
    assert [p.side for p in pages] == ["A", "B"] * 3


def test_graceful_stop_lets_the_feeding_stack_finish():
    """Stop mid-stack: don't cancel (the scanner has bills inside); the op
    ends when the feeder empties and every page is kept."""
    from dollar_detective.scanner.backend import StopEvent
    op, folder = FakeOp(), _folder()
    _writer(folder, op, 40, gap=0.01)
    stop = StopEvent()
    n = 0
    for _ in _pump(op, folder, stop=stop):
        n += 1
        if n == 6:
            stop.set()
    assert not op.cancelled and n == 40
    assert not folder.exists()


def test_stop_now_cancels_but_keeps_pages_on_disk():
    from dollar_detective.scanner.backend import StopEvent
    op, folder = FakeOp(), _folder()
    _writer(folder, op, 200, gap=0.02)
    stop = StopEvent()
    n = 0
    for _ in _pump(op, folder, stop=stop):
        n += 1
        if n == 6:
            stop.set_now()
    assert op.cancelled
    assert n >= 6                          # nothing already on disk is lost
    assert not folder.exists()


def test_error_keeps_pages_then_raises_mapped():
    op, folder = FakeOp(), _folder()
    _writer(folder, op, 4, end_status=3, error_code=0x80210002)
    got = []
    with pytest.raises(ScanError) as ei:
        for p in _pump(op, folder):
            got.append(p.side)
    assert got == ["A", "B", "A", "B"]
    assert ei.value.cause == "JAM"


def test_empty_feeder_error_maps_to_PE():
    op, folder = FakeOp(), _folder()
    op.error_code, op._status = 0x80210003, 3
    with pytest.raises(ScanError) as ei:
        list(_pump(op, folder))
    assert ei.value.cause == "PE"


def test_consumer_abandoning_cancels_op():
    op, folder = FakeOp(), _folder()
    _writer(folder, op, 200, gap=0.02)
    gen = _pump(op, folder)
    next(gen)
    gen.close()
    assert op.cancelled


def test_unknown_device_falls_back_to_first():
    b = object.__new__(wb.WinRtScannerBackend)
    b.list_devices = lambda: [ScannerDevice(id="a", name="A"), ScannerDevice(id="b", name="B")]
    assert b._resolve_device("b") == "b"
    assert b._resolve_device("zzz") == "a"
    assert b._resolve_device(None) == "a"
    b.list_devices = lambda: []
    with pytest.raises(ScanError):
        b._resolve_device(None)
