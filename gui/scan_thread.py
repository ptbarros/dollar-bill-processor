"""Background worker that drives a feed scanner and drops finished bill images
into a folder, where the normal Monitor / live pipeline picks them up.

Mirrors the file naming the pipeline already understands -- FastFoto_NNNNN.jpg
(front) + FastFoto_NNNNN_b.jpg (back) -- and writes each file under a temp name
then renames it, so the folder watcher never sees a half-written image. Running
on a QThread keeps the ~1.5s/page feed off the GUI thread.
"""

import os
import re
import threading
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal

from dollar_detective.debug_logger import dlog

_NUM_RE = re.compile(r"^FastFoto_(\d{5})(?:_b)?\.jpg$", re.IGNORECASE)


class ScanThread(QThread):
    """Drives `backend.scan_pages()` and writes pages into `out_dir`."""

    page_written = Signal(int)     # total sides written so far
    status = Signal(str)           # human-readable progress line
    finished_ok = Signal(int)      # total sides written
    failed = Signal(str)           # error message

    def __init__(self, backend, out_dir, *, dpi=300, duplex=True, color=True,
                 device_id=None, start_number=None, keep_armed=True,
                 options=None, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.out_dir = Path(out_dir)
        self.dpi = dpi
        self.duplex = duplex
        self.color = color
        self.device_id = device_id
        # Caller-supplied starting number keeps numbering monotonic across
        # successive scans in one session, even after the live pipeline has
        # drained earlier files out of the folder. None -> derive from the folder.
        self.start_number = start_number
        # Highest number assigned this run (read by the caller to continue).
        self.final_number = (start_number - 1) if start_number else 0
        # Stay armed between feeder loads (False = one pass, then finish).
        self.keep_armed = keep_armed
        # Backend-specific choices, e.g. {"driver": "auto"|"twain"|"wia"}.
        self.options = dict(options or {})
        self._stop = threading.Event()
        self._bills = 0
        self._sides = 0

    def request_stop(self):
        self._stop.set()

    def _next_number(self) -> int:
        """Continue FastFoto numbering after any files already in the folder."""
        hi = 0
        try:
            for f in self.out_dir.iterdir():
                m = _NUM_RE.match(f.name)
                if m:
                    hi = max(hi, int(m.group(1)))
        except Exception:
            pass
        return hi + 1

    def _write(self, name: str, image) -> None:
        """Atomically write a JPEG so a folder watcher only ever sees it whole.
        Encode explicitly (not cv2.imwrite) so the temp file's extension doesn't
        have to be .jpg for OpenCV to pick the encoder."""
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        final = self.out_dir / name
        tmp = self.out_dir / ("." + name + ".tmp")
        tmp.write_bytes(buf.tobytes())
        os.replace(str(tmp), str(final))

    def _one_pass(self, number: int):
        """Scan whatever is in the feeder now. Returns (next_number, sides
        written). Raises the backend's ScanError."""
        written = 0
        # Pair by POSITION, not arrival order: the scanner doesn't always send
        # a bill's front before its back (duplex sides can arrive batched), so
        # the nth front goes with the nth back. Assuming strict A,B order made
        # backs overwrite the wrong front's file and silently drop bills.
        fronts, backs = [], []
        try:
            # Don't break out on Stop: the backend cancels the feed itself and
            # still hands over every page it already scanned.
            for page in self.backend.scan_pages(
                    self.device_id, dpi=self.dpi, duplex=self.duplex,
                    color=self.color, stop=self._stop, **self.options):
                (backs if page.side == "B" else fronts).append(page.image)
                while fronts and backs:
                    self._write(f"FastFoto_{number:05d}.jpg", fronts.pop(0))
                    self._write(f"FastFoto_{number:05d}_b.jpg", backs.pop(0))
                    number += 1
                    written += 2
                    self._bills += 1
                    self._sides += 2
                    self.page_written.emit(self._sides)
                    self.status.emit(
                        f"Scanned {self._bills} bill{'s' if self._bills != 1 else ''}…")
        finally:
            # Leftover fronts with no back (simplex, or a feed that stopped
            # mid-bill): save them front-only. Orphan backs are dropped.
            for front in fronts:
                self._write(f"FastFoto_{number:05d}.jpg", front)
                number += 1
                written += 1
                self._bills += 1
                self._sides += 1
            # Record progress even if the pass failed, so the next Scan never
            # reuses a number the pipeline may already have taken.
            self.final_number = number - 1
        return number, written

    def run(self):
        """Scan stack after stack until Stop. Between stacks the session stays
        armed -- reload the feeder and it carries on, no click needed (clicking
        between loads is how whole stacks got cancelled by mistake)."""
        self._bills = 0
        self._sides = 0
        empty_reported = False
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            # Continue from the caller's counter when given (monotonic across
            # scans in a session, even after the pipeline drained the folder);
            # otherwise derive from whatever's in the folder.
            number = self.start_number if self.start_number else self._next_number()
            self.final_number = number - 1
            self.status.emit("Feed your strap into the scanner…")
            while not self._stop.is_set():
                try:
                    number, got = self._one_pass(number)
                except Exception as e:
                    # An empty feeder isn't an error, just nothing to scan: the
                    # FF-640's direct driver reports it as a "jam" (PJ), WIA as
                    # paper-empty (PE).
                    if getattr(e, "cause", "") not in ("PJ", "PE"):
                        raise
                    got = 0
                dlog("scan.pass_end", sides=got, bills_total=self._bills,
                     final_number=self.final_number)
                if self._stop.is_set() or not self.keep_armed:
                    break
                if got or not empty_reported:
                    empty_reported = True
                    done = (f"{self._bills} bill{'s' if self._bills != 1 else ''} scanned. "
                            if self._bills else "")
                    self.status.emit(
                        f"{done}Waiting for bills — load the next stack and it "
                        "scans automatically. Click Stop Scan when the strap is done.")
                # Right after a stack that fed, go again straight away: bills
                # dropped in while it was still feeding aren't picked up by that
                # pass (the scanner only takes what was loaded when it started),
                # so they're already waiting. An empty pass costs ~0.1 s.
                if got:
                    continue
                if not self.backend.wait_for_paper(self.device_id, self._stop):
                    break
            self.finished_ok.emit(self._sides)
        except Exception as e:
            self.failed.emit(str(e))
