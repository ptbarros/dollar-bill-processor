"""Direct feed-scanner control for Dollar Detective.

A small backend abstraction over the OS/vendor scanning APIs so an in-app "Scan"
button can drive the feeder directly -- acquiring each bill as it feeds and
handing it to the normal pipeline -- instead of the user opening the scanner's
own software and watching a folder.

Scanning has no cross-platform API: Windows has WIA/TWAIN, macOS has
ImageCaptureCore, Linux has SANE (or, for scanners it doesn't cover well, a
direct vendor-protocol driver). So each platform gets its own ScannerBackend;
``get_scanner_backend()`` returns the one that can drive a connected scanner on
this machine, or None.

Phase 1 ships the Linux Epson ESC/I-2 backend for the FastFoto FF-640 (Paul's
own `esci2.py` driver, vendored). The interface is deliberately the same shape a
WiaBackend / SaneBackend would implement, so later platforms slot in without the
GUI changing.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class ScannerDevice:
    """A scanner the backend can open."""
    id: str
    name: str


@dataclass
class ScanPage:
    """One finished side off the feeder. `side` is 'A' (front) or 'B' (back);
    `image` is a BGR numpy array (already cropped/straightened when the backend
    does that itself)."""
    side: str
    image: np.ndarray


class ScannerBackend(ABC):
    """One platform's feed-scanner driver."""

    #: Short name shown in the UI, e.g. "Epson FastFoto (direct)".
    label: str = "Scanner"

    @abstractmethod
    def list_devices(self) -> List[ScannerDevice]:
        """Scanners this backend can currently open (empty if none connected)."""

    @abstractmethod
    def scan_pages(self, device_id: Optional[str] = None, *,
                   dpi: int = 300, duplex: bool = True, color: bool = True,
                   stop=None) -> Iterator[ScanPage]:
        """Feed and scan every sheet in the feeder, yielding one ScanPage per
        finished side as it arrives (so the caller can process live). `stop` is
        an optional threading.Event to cancel mid-feed. Raises ScanError."""

    def feeder_has_paper(self, device_id: Optional[str] = None) -> bool:
        """Quick read of whether paper is loaded in the feeder (for auto-start).
        Best-effort: False on any error or when unsupported. Must not be called
        while a scan is in progress (it opens its own device session)."""
        return False


class ScanError(Exception):
    """A scan failed (device busy, jam, offline, ...). `cause` carries the raw
    scanner code where known, e.g. 'PJ' (jam / empty feeder), 'PE' (paper empty)."""

    def __init__(self, msg, cause=""):
        super().__init__(msg)
        self.cause = cause


# --------------------------------------------------------------------------- #
# Linux: Epson ESC/I-2 direct (FastFoto FF-640)
# --------------------------------------------------------------------------- #
class EpsonEsciBackend(ScannerBackend):
    """Drives the Epson FastFoto FF-640 directly over USB via the vendored
    ESC/I-2 driver, cropping and straightening each page like the standalone
    FastFoto Scan app does. Pages stream off the feeder one side at a time."""

    label = "Epson FastFoto (direct)"

    def __init__(self):
        # Import lazily so a machine without pyusb / without the scanner still
        # loads the app fine; availability is decided by list_devices().
        from . import esci2
        self._esci2 = esci2

    def list_devices(self) -> List[ScannerDevice]:
        try:
            import usb.core
            dev = usb.core.find(idVendor=self._esci2.VENDOR,
                                idProduct=self._esci2.PRODUCT)
        except Exception:
            return []
        if dev is None:
            return []
        return [ScannerDevice(id="ff640", name="Epson FastFoto FF-640")]

    def feeder_has_paper(self, device_id: Optional[str] = None) -> bool:
        try:
            with self._esci2.FF640() as scanner:
                return bool(scanner.paper_loaded())
        except Exception:
            return False

    # Bills feed long-edge-first, so only ~66mm travels; 120mm is a generous
    # window (matches the standalone FastFoto app) and far faster than the 14"
    # default. "Normal" crop margin keeps a little background.
    SCAN_LENGTH_MM = 120
    MARGIN_PX = 6

    def scan_pages(self, device_id: Optional[str] = None, *,
                   dpi: int = 300, duplex: bool = True, color: bool = True,
                   stop=None) -> Iterator[ScanPage]:
        from . import billcrop
        max_len_px = int(self.SCAN_LENGTH_MM / 25.4 * dpi)
        try:
            with self._esci2.FF640() as scanner:
                for side, img in scanner.scan(dpi=dpi, duplex=duplex, color=color,
                                              overscan=True, max_len_px=max_len_px,
                                              stop=stop):
                    try:
                        cropped, _found = billcrop.crop_item(img, margin_px=self.MARGIN_PX)
                    except Exception:
                        cropped = img   # never drop a page over a crop hiccup
                    yield ScanPage(side=side, image=cropped)
        except self._esci2.ScanError as e:
            # Preserve the raw cause so the caller can tell an empty feeder (PJ
            # before any page) from a real jam.
            raise ScanError(str(e), cause=getattr(e, "cause", "")) from e


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def available_backends() -> List[ScannerBackend]:
    """Every backend that could run on this OS (not necessarily with a scanner
    plugged in). Ordered best-first."""
    out: List[ScannerBackend] = []
    if sys.platform.startswith("linux"):
        try:
            out.append(EpsonEsciBackend())
        except Exception:
            pass
    # Future: WiaBackend/TwainBackend on win32, ImageCaptureBackend on darwin.
    return out


def get_scanner_backend() -> Tuple[Optional[ScannerBackend], List[ScannerDevice]]:
    """The backend with a scanner connected right now, plus its device list.
    Returns (None, []) when nothing is connected / supported. Cheap and
    exception-safe -- fine to call from the GUI to decide whether to show the
    Scan button."""
    for backend in available_backends():
        try:
            devices = backend.list_devices()
        except Exception:
            devices = []
        if devices:
            return backend, devices
    return None, []
