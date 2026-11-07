"""Windows feed-scanner backend: the WinRT ImageScanner API (WIA 2.0).

Why WinRT and not classic WIA scripting (WIA.DeviceManager / wiaaut.dll): on the
Epson FF-640 the scripting layer can't return duplex at all (E_OUTOFMEMORY after
the bill feeds) and only takes one page per request, so the feeder stops and
restarts between bills (~2.9 s/bill, fronts only). WinRT's ImageScanner is
Microsoft's own WIA 2.0 client: duplex works, the feeder runs continuously, and
each side lands as its bill feeds (measured 2026-11-07: 12 bills -> 24 JPEGs,
strictly front,back,front,back, ~1.1 s/bill). Any scanner with a WIA driver --
including FIL's Canon R40 -- goes through the same API.

ScanFilesToFolderAsync writes the pages itself, so we point it at a private temp
folder, read each new file as the progress callback reports it, crop it, and
hand it on as a ScanPage. The caller (ScanThread) then writes the app's usual
FastFoto_NNNNN.jpg / _b.jpg names into the Monitor folder.
"""

from __future__ import annotations

import asyncio
import queue
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np

from .backend import ScanError, ScannerBackend, ScanPage, ScannerDevice

# WIA HRESULTs worth recognising (see wia_lh.h).
_WIA_PAPER_EMPTY = 0x80210003
_WIA_PAPER_JAM = 0x80210002
_WIA_BUSY = 0x80210006
_WIA_OFFLINE = 0x80210005

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".bmp", ".png", ".tif", ".tiff"}


def _hresult(exc) -> int:
    code = getattr(exc, "winerror", None)
    if code is None and exc.args and isinstance(exc.args[0], int):
        code = exc.args[0]
    return (code or 0) & 0xFFFFFFFF


def _run_sync(coro_fn, *args):
    """Run an async function to completion on a private thread + event loop, so
    it's safe from the GUI thread (no loop running there) and from QThreads."""
    box = {}

    def runner():
        try:
            box["value"] = asyncio.run(coro_fn(*args))
        except BaseException as e:   # noqa: BLE001 -- re-raised below
            box["error"] = e

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


async def _await(op):
    return await op


def new_page_files(folder: Path, seen: set) -> List[Path]:
    """Image files in `folder` not yet in `seen`, oldest first.

    The scanner names pages SCN_<date>.jpg, SCN_<date>_2.jpg, ... so creation
    time is the reliable order; name length then name breaks ties (puts _2
    before _10)."""
    fresh = []
    try:
        for p in folder.iterdir():
            if p.suffix.lower() in _IMAGE_SUFFIXES and p.name not in seen:
                try:
                    ctime = p.stat().st_ctime
                except OSError:
                    continue
                fresh.append((ctime, len(p.stem), p.name, p))
    except OSError:
        return []
    fresh.sort()
    return [t[3] for t in fresh]


def _read_image(path: Path, timeout: float = 3.0):
    """Decode an image the scanner just wrote, retrying briefly in case it's
    still being flushed."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is not None or time.monotonic() > deadline:
            return img
        time.sleep(0.05)


class WinRtScannerBackend(ScannerBackend):
    """Any WIA feed scanner on Windows 10/11, via Windows.Devices.Scanners."""

    label = "Windows scanner (WIA)"
    # WinRT has no "paper in feeder" query, so auto-start isn't offered.
    can_detect_paper = False

    # Bills feed long-edge-first, so only ~2.6" of paper travels; 4.72" (120mm)
    # is generous headroom for skew and matches the Linux backend's window.
    SCAN_LENGTH_IN = 4.72
    MARGIN_PX = 6

    def __init__(self):
        # Fail fast (-> backend not offered) when the WinRT projection isn't
        # installed, e.g. a dev venv without the winrt-* packages.
        from winrt.windows.devices import scanners  # noqa: F401
        from winrt.windows.devices import enumeration  # noqa: F401

    # -- discovery -----------------------------------------------------------
    @staticmethod
    async def _find_devices():
        from winrt.windows.devices.enumeration import DeviceInformation
        from winrt.windows.devices.scanners import ImageScanner

        selector = ImageScanner.get_device_selector()
        finder = (getattr(DeviceInformation, "find_all_async_aqs_filter", None)
                  or DeviceInformation.find_all_async)
        infos = await finder(selector)
        return [(i.id, i.name) for i in infos]

    def list_devices(self) -> List[ScannerDevice]:
        try:
            return [ScannerDevice(id=i, name=n) for i, n in _run_sync(self._find_devices)]
        except Exception:
            return []

    def _resolve_device(self, device_id: Optional[str]) -> str:
        devices = self.list_devices()
        if not devices:
            raise ScanError("No scanner found. Is it plugged in and switched on?", cause="OFFLINE")
        for d in devices:
            if device_id and d.id == device_id:
                return d.id
        return devices[0].id

    # -- scanning ------------------------------------------------------------
    async def _scan(self, device_id, folder: Path, dpi, duplex, color, events, stop):
        from winrt.windows.devices.scanners import (
            ImageScanner, ImageScannerColorMode, ImageScannerFormat,
            ImageScannerResolution, ImageScannerScanSource)
        from winrt.windows.storage import StorageFolder

        scanner = await ImageScanner.from_id_async(device_id)
        if not scanner.is_scan_source_supported(ImageScannerScanSource.FEEDER):
            raise ScanError("This scanner has no document feeder.", cause="NOFEEDER")
        cfg = scanner.feeder_configuration

        cfg.duplex = bool(duplex and cfg.can_scan_duplex)
        events.put(("duplex", cfg.duplex))
        cfg.max_number_of_pages = 0          # everything in the feeder
        cfg.color_mode = (ImageScannerColorMode.COLOR if color
                          else ImageScannerColorMode.GRAYSCALE)
        if cfg.is_format_supported(ImageScannerFormat.JPEG):
            cfg.format = ImageScannerFormat.JPEG
        cfg.desired_resolution = ImageScannerResolution(dpi_x=float(dpi), dpi_y=float(dpi))

        # Shorten the scan window to bill length (faster, smaller files). Full
        # feed width keeps the scanner's grey backing on both sides of the bill,
        # which is what the cropper keys off. Best-effort: drivers may refuse.
        try:
            from winrt.windows.graphics.printing import PrintMediaSize
            cfg.page_size = PrintMediaSize.PRINTER_CUSTOM
        except Exception:
            pass
        try:
            from winrt.windows.foundation import Rect
            area = cfg.max_scan_area
            cfg.selected_scan_region = Rect(
                0, 0, area.width, min(area.height, self.SCAN_LENGTH_IN))
        except Exception:
            pass

        target = await StorageFolder.get_folder_from_path_async(str(folder))
        op = scanner.scan_files_to_folder_async(ImageScannerScanSource.FEEDER, target)
        op.progress = lambda _op, n: events.put(("page", int(n)))
        task = asyncio.ensure_future(_await(op))
        while not task.done():
            if stop is not None and stop.is_set():
                try:
                    op.cancel()
                except Exception:
                    pass
                break
            await asyncio.sleep(0.1)
        try:
            await task
        except Exception:
            if stop is not None and stop.is_set():
                return   # cancelled on purpose
            raise

    def scan_pages(self, device_id: Optional[str] = None, *,
                   dpi: int = 300, duplex: bool = True, color: bool = True,
                   stop=None) -> Iterator[ScanPage]:
        from . import billcrop

        dev = self._resolve_device(device_id)
        folder = Path(tempfile.mkdtemp(prefix="dd_scan_"))
        events: "queue.Queue" = queue.Queue()
        stop = stop or threading.Event()

        def worker():
            try:
                asyncio.run(self._scan(dev, folder, dpi, duplex, color, events, stop))
            except BaseException as e:   # noqa: BLE001 -- forwarded to caller
                events.put(("error", e))
            finally:
                events.put(("done", None))

        thread = threading.Thread(target=worker, name="winrt-scan", daemon=True)
        thread.start()

        seen: set = set()
        index = 0            # pages handed out so far
        is_duplex = duplex

        def drain():
            nonlocal index
            for path in new_page_files(folder, seen):
                seen.add(path.name)
                img = _read_image(path)
                if img is None:
                    continue
                index += 1
                side = "B" if (is_duplex and index % 2 == 0) else "A"
                try:
                    cropped, _found = billcrop.crop_item(img, margin_px=self.MARGIN_PX)
                except Exception:
                    cropped = img   # never drop a page over a crop hiccup
                yield ScanPage(side=side, image=cropped)

        try:
            while True:
                kind, value = events.get()
                if kind == "duplex":
                    is_duplex = bool(value)
                elif kind == "page":
                    yield from drain()
                elif kind == "error":
                    yield from drain()   # keep whatever made it before the error
                    code = _hresult(value)
                    if isinstance(value, ScanError):
                        raise value
                    if code == _WIA_PAPER_EMPTY:
                        raise ScanError("No paper in the feeder.", cause="PE") from value
                    if code == _WIA_PAPER_JAM:
                        raise ScanError("Paper jam — clear the feeder and try again.",
                                        cause="JAM") from value
                    if code == _WIA_BUSY:
                        raise ScanError("The scanner is busy (is another scanning app "
                                        "open?).", cause="BUSY") from value
                    if code == _WIA_OFFLINE:
                        raise ScanError("The scanner is offline.", cause="OFFLINE") from value
                    raise ScanError(f"Scanner error: {value}", cause=f"0x{code:08X}") from value
                elif kind == "done":
                    yield from drain()   # anything the progress callback missed
                    return
        finally:
            stop.set()
            thread.join(timeout=10)
            shutil.rmtree(folder, ignore_errors=True)
