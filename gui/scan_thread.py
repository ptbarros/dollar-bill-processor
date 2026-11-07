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

_NUM_RE = re.compile(r"^FastFoto_(\d{5})(?:_b)?\.jpg$", re.IGNORECASE)


class ScanThread(QThread):
    """Drives `backend.scan_pages()` and writes pages into `out_dir`."""

    page_written = Signal(int)     # total sides written so far
    status = Signal(str)           # human-readable progress line
    finished_ok = Signal(int)      # total sides written
    failed = Signal(str)           # error message

    def __init__(self, backend, out_dir, *, dpi=300, duplex=True, color=True,
                 device_id=None, start_number=None, parent=None):
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
        self._stop = threading.Event()

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

    def run(self):
        written = 0
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            # Continue from the caller's counter when given (monotonic across
            # scans in a session, even after the pipeline drained the folder);
            # otherwise derive from whatever's in the folder.
            number = self.start_number if self.start_number else self._next_number()
            cur_front = None   # number assigned to the most recent front (side A)
            self.status.emit("Feed your strap into the scanner…")
            for page in self.backend.scan_pages(
                    self.device_id, dpi=self.dpi, duplex=self.duplex,
                    color=self.color, stop=self._stop):
                if page.side == "A":
                    cur_front = number
                    self._write(f"FastFoto_{number:05d}.jpg", page.image)
                    number += 1
                else:  # 'B' -> back of the most recent front (or its own number)
                    n = cur_front if cur_front is not None else number
                    if cur_front is None:
                        number += 1
                    self._write(f"FastFoto_{n:05d}_b.jpg", page.image)
                written += 1
                self.page_written.emit(written)
                bills = (written + 1) // 2 if self.duplex else written
                self.status.emit(f"Scanned {bills} bill{'s' if bills != 1 else ''}…")
                if self._stop.is_set():
                    break
            self.final_number = number - 1   # highest number actually assigned
            self.finished_ok.emit(written)
        except Exception as e:
            # The FF-640 reports a "jam" (PJ) with no pages when the feeder is
            # simply empty -- that's not an error, just nothing to scan.
            cause = getattr(e, "cause", "")
            if cause == "PJ" and written == 0:
                self.status.emit("No bills in the feeder — load a strap and click Scan.")
                self.finished_ok.emit(0)
            else:
                self.failed.emit(str(e))
