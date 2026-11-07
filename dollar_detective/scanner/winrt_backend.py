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
folder and hand each finished file on as a ScanPage; ScanThread then writes the
app's usual FastFoto_NNNNN.jpg / _b.jpg names into the Monitor folder.

Robustness rules (each learned from a real failure on the test laptop):
- Never attach Progress/Completed handlers. The scan op can come back before
  it's "started", and attaching a handler then throws 0x80000019
  (E_ASYNC_OPERATION_NOT_STARTED) -- which killed our reader while the scanner
  kept feeding a whole stack into a folder nobody read. We poll op.status and
  the folder instead.
- A page is handed on only once it's complete: a newer file exists, or its size
  has held steady, or the op has finished.
- Front/back comes from the file's POSITION in scan order (odd = front, even =
  back), so one unreadable file can't flip every later pairing.
- Stop cancels the feed but still hands on every page already scanned.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np

from .backend import ScanError, ScannerBackend, ScanPage, ScannerDevice

try:
    from dollar_detective.debug_logger import dlog
except Exception:   # pragma: no cover -- logging must never break scanning
    def dlog(*_a, **_k):
        pass

# HRESULTs worth recognising (wia_lh.h / winerror.h).
_WIA_PAPER_EMPTY = 0x80210003
_WIA_PAPER_JAM = 0x80210002
_WIA_OFFLINE = 0x80210005
_WIA_BUSY = 0x80210006
_E_ASYNC_NOT_STARTED = 0x80000019

_STARTED, _COMPLETED, _CANCELED, _ERROR = 0, 1, 2, 3

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".bmp", ".png", ".tif", ".tiff"}

POLL_S = 0.1            # folder/status poll interval while scanning
STEADY_S = 0.6          # a last file whose size holds this long is complete
READ_RETRY_S = 5.0      # how long to keep retrying an unreadable finished file


def _hresult(exc) -> int:
    code = getattr(exc, "winerror", None)
    if code is None and getattr(exc, "args", None) and isinstance(exc.args[0], int):
        code = exc.args[0]
    return int(code or 0) & 0xFFFFFFFF


class _OpError(Exception):
    def __init__(self, code: int):
        super().__init__(f"0x{code:08X}")
        self.code = code


def _status(op) -> int:
    """op.status, treating 'not started yet' as still running."""
    try:
        return int(op.status)
    except OSError as e:
        if _hresult(e) == _E_ASYNC_NOT_STARTED:
            return _STARTED
        raise


def _wait(op, timeout: float = 30.0):
    """Block until a WinRT async op finishes (by polling -- no handlers)."""
    deadline = time.monotonic() + timeout
    while True:
        st = _status(op)
        if st == _COMPLETED:
            return op.get_results()
        if st == _ERROR:
            raise _OpError(int(op.error_code) & 0xFFFFFFFF)
        if st == _CANCELED:
            return None
        if time.monotonic() > deadline:
            try:
                op.cancel()
            except Exception:
                pass
            raise ScanError("The scanner didn't respond.", cause="TIMEOUT")
        time.sleep(0.02)


def ordered_page_files(folder: Path) -> List[Path]:
    """Every image file in `folder`, in scan order.

    The scanner names pages SCN_<date>.jpg, SCN_<date>_2.jpg, ... so creation
    time is the reliable order; name length then name break ties (puts _2
    before _10)."""
    out = []
    try:
        for p in folder.iterdir():
            if p.suffix.lower() in _IMAGE_SUFFIXES and not p.name.startswith("."):
                try:
                    ctime = p.stat().st_ctime
                except OSError:
                    continue
                out.append((ctime, len(p.stem), p.name, p))
    except OSError:
        return []
    out.sort()
    return [t[3] for t in out]


def _decode(path: Path):
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def _map_error(code: int) -> ScanError:
    if code == _WIA_PAPER_EMPTY:
        return ScanError("No paper in the feeder.", cause="PE")
    if code == _WIA_PAPER_JAM:
        return ScanError("Paper jam — clear the feeder and try again.", cause="JAM")
    if code == _WIA_BUSY:
        return ScanError("The scanner is busy (is another scanning app open?).", cause="BUSY")
    if code == _WIA_OFFLINE:
        return ScanError("The scanner is offline.", cause="OFFLINE")
    return ScanError(f"Scanner error 0x{code:08X}", cause=f"0x{code:08X}")


class WinRtScannerBackend(ScannerBackend):
    """Any WIA feed scanner on Windows 10/11, via Windows.Devices.Scanners."""

    label = "Windows scanner (WIA)"
    # WinRT has no "paper in feeder" query, so auto-start isn't offered.
    can_detect_paper = False

    # Scan-window length. On the Epson FF-640 bills feed long-edge-first, so
    # only ~2.6" of paper travels and 4.72" (120mm) is generous headroom
    # (matches the Linux backend). Other scanners get their FULL length until
    # the Scanner Test has measured them -- a bill fed short-edge-first is
    # 6.14" long and a short window would cut every one off.
    SCAN_LENGTH_IN = 4.72
    KNOWN_SHORT_FEED = ("VID_04B8&PID_015F",)    # Epson FastFoto FF-640
    MARGIN_PX = 6

    @classmethod
    def default_scan_length(cls, device_id: str) -> Optional[float]:
        """Scan length (inches) to use for a device, or None = full length."""
        up = (device_id or "").upper()
        return cls.SCAN_LENGTH_IN if any(k in up for k in cls.KNOWN_SHORT_FEED) else None

    # Transfer format. The FF-640 feeds ~0.55 s/bill but compresses JPEG at only
    # ~1.12 s/bill, so it buffers and the feeder empties ~0.57 s x N before the
    # scan ends. Uncompressed (DIB) delivers at ~0.78 s/bill over USB 2 (A/B on
    # Paul's laptop, 2026-11-07); we JPEG-encode on our side anyway. Other
    # scanners keep JPEG until measured.
    UNCOMPRESSED_FASTER = ("VID_04B8&PID_015F",)

    # Scan-window WIDTH for the FF-640. Uncompressed transfer is USB-bound
    # (~21 MB/s), so time scales with width: 8.5" = 1.02 s/bill, 6.5" = 0.78.
    # 7.0" leaves ~0.4" of grey backing each side of a 6.14" bill (the driver
    # centres the window) so a crooked bill doesn't touch the edges the
    # cropper samples, for ~0.06 s/bill more than 6.5".
    SCAN_WIDTH_IN = 7.0

    @classmethod
    def default_scan_width(cls, device_id: str) -> Optional[float]:
        up = (device_id or "").upper()
        return cls.SCAN_WIDTH_IN if any(k in up for k in cls.KNOWN_SHORT_FEED) else None

    @classmethod
    def default_format(cls, device_id: str) -> str:
        up = (device_id or "").upper()
        return "dib" if any(k in up for k in cls.UNCOMPRESSED_FASTER) else "jpeg"

    def __init__(self):
        # Fail fast (-> backend not offered) when the WinRT projection isn't
        # installed, e.g. a dev venv without the winrt-* packages.
        from winrt.windows.devices import scanners  # noqa: F401
        from winrt.windows.devices import enumeration  # noqa: F401

    # -- discovery -----------------------------------------------------------
    def list_devices(self) -> List[ScannerDevice]:
        try:
            from winrt.windows.devices.enumeration import DeviceInformation
            from winrt.windows.devices.scanners import ImageScanner
            infos = _wait(DeviceInformation.find_all_async_aqs_filter(
                ImageScanner.get_device_selector()), timeout=10)
            return [ScannerDevice(id=i.id, name=i.name) for i in (infos or [])]
        except Exception as e:
            dlog("scan.winrt.list_failed", error=repr(e))
            return []

    def _resolve_device(self, device_id: Optional[str]) -> str:
        devices = self.list_devices()
        if not devices:
            raise ScanError("No scanner found. Is it plugged in and switched on?", cause="OFFLINE")
        for d in devices:
            if device_id and d.id == device_id:
                return d.id
        return devices[0].id

    def describe(self, device_id: Optional[str] = None) -> dict:
        """Capabilities as WinRT reports them (no scanning)."""
        from winrt.windows.devices.scanners import (
            ImageScanner, ImageScannerAutoCroppingMode, ImageScannerColorMode,
            ImageScannerFormat, ImageScannerScanSource)
        info = super().describe(device_id)
        dev = self._resolve_device(device_id)
        info["id"] = dev
        info["name"] = next((d.name for d in self.list_devices() if d.id == dev), "")
        info["default_scan_length_in"] = self.default_scan_length(dev)
        sc = _wait(ImageScanner.from_id_async(dev), timeout=15)

        def safe(fn, default=None):
            try:
                return fn()
            except Exception as e:
                return f"error: {e}"

        info["default_source"] = safe(lambda: ImageScannerScanSource(sc.default_scan_source).name)
        info["sources"] = {s.name: safe(lambda s=s: bool(sc.is_scan_source_supported(s)))
                           for s in ImageScannerScanSource}
        if info["sources"].get("FEEDER") is True:
            cfg = sc.feeder_configuration
            res = lambda r: [round(r.dpi_x), round(r.dpi_y)]
            size = lambda z: [round(z.width, 2), round(z.height, 2)]
            info["feeder"] = {
                "can_scan_duplex": safe(lambda: bool(cfg.can_scan_duplex)),
                "can_scan_ahead": safe(lambda: bool(cfg.can_scan_ahead)),
                "can_auto_detect_page_size": safe(lambda: bool(cfg.can_auto_detect_page_size)),
                "max_number_of_pages": safe(lambda: int(cfg.max_number_of_pages)),
                "resolution_min": safe(lambda: res(cfg.min_resolution)),
                "resolution_max": safe(lambda: res(cfg.max_resolution)),
                "resolution_optical": safe(lambda: res(cfg.optical_resolution)),
                "scan_area_min_in": safe(lambda: size(cfg.min_scan_area)),
                "scan_area_max_in": safe(lambda: size(cfg.max_scan_area)),
                "formats": safe(lambda: [f.name for f in ImageScannerFormat
                                         if cfg.is_format_supported(f)]),
                "color_modes": safe(lambda: [m.name for m in ImageScannerColorMode
                                             if cfg.is_color_mode_supported(m)]),
                "auto_crop_modes": safe(lambda: [m.name for m in ImageScannerAutoCroppingMode
                                                 if cfg.is_auto_cropping_mode_supported(m)]),
            }
        return info

    def _configure(self, scanner, dpi, duplex, color, scan_length_in=None,
                   transfer_format="jpeg", scan_width_in=None) -> bool:
        from winrt.windows.devices.scanners import (
            ImageScannerColorMode, ImageScannerFormat, ImageScannerResolution,
            ImageScannerScanSource)
        if not scanner.is_scan_source_supported(ImageScannerScanSource.FEEDER):
            raise ScanError("This scanner has no document feeder.", cause="NOFEEDER")
        cfg = scanner.feeder_configuration
        cfg.duplex = bool(duplex and cfg.can_scan_duplex)
        cfg.max_number_of_pages = 0          # everything in the feeder
        cfg.color_mode = (ImageScannerColorMode.COLOR if color
                          else ImageScannerColorMode.GRAYSCALE)
        fmt = (ImageScannerFormat.DEVICE_INDEPENDENT_BITMAP if transfer_format == "dib"
               else ImageScannerFormat.JPEG)
        if cfg.is_format_supported(fmt):
            cfg.format = fmt
        elif cfg.is_format_supported(ImageScannerFormat.JPEG):
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
            length = area.height if not scan_length_in else min(area.height, scan_length_in)
            width = area.width if not scan_width_in else min(area.width, scan_width_in)
            # X=0: drivers centre a narrower window in the feed path themselves.
            cfg.selected_scan_region = Rect(0, 0, width, length)
        except Exception:
            pass
        return bool(cfg.duplex)

    # -- scanning ------------------------------------------------------------
    def scan_pages(self, device_id: Optional[str] = None, *,
                   dpi: int = 300, duplex: bool = True, color: bool = True,
                   stop=None, scan_length_in="auto", transfer_format="auto",
                   scan_width_in="auto", **options) -> Iterator[ScanPage]:
        """`scan_length_in`: "auto" (FF-640 short window, else full length),
        None (full length), or a length in inches (Scanner Test).
        `transfer_format`: "auto" (per-scanner default), "jpeg" or "dib"."""
        from winrt.windows.devices.scanners import ImageScanner, ImageScannerScanSource
        from winrt.windows.storage import StorageFolder

        try:
            dev = self._resolve_device(device_id)
            scanner = _wait(ImageScanner.from_id_async(dev), timeout=15)
            if scan_length_in == "auto":
                scan_length_in = self.default_scan_length(dev)
            if transfer_format == "auto":
                transfer_format = self.default_format(dev)
            if scan_width_in == "auto":
                scan_width_in = self.default_scan_width(dev)
            is_duplex = self._configure(scanner, dpi, duplex, color, scan_length_in,
                                        transfer_format, scan_width_in)
        except _OpError as e:
            raise _map_error(e.code) from e

        folder = Path(tempfile.mkdtemp(prefix="dd_scan_"))
        try:
            target = _wait(StorageFolder.get_folder_from_path_async(str(folder)), timeout=10)
            op = scanner.scan_files_to_folder_async(ImageScannerScanSource.FEEDER, target)
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        t0 = time.monotonic()
        dlog("scan.winrt.start", folder=str(folder), dpi=dpi, duplex=is_duplex,
             scan_length_in=scan_length_in, transfer_format=transfer_format,
             scan_width_in=scan_width_in)
        yield from self._pump(op, folder, is_duplex, stop, t0)

    def _pump(self, op, folder: Path, is_duplex: bool, stop, t0: float) -> Iterator[ScanPage]:
        """Hand on finished pages until the op ends; then clean up."""
        from . import billcrop

        handed = 0                    # files handed on (by position)
        last_size = {}                # name -> (size, since) for the newest file
        unreadable_since = {}         # name -> first failed read time
        cancelled = False
        final_status = _STARTED
        error_code = 0

        def ready_count(files, finished) -> int:
            """How many files (from the front of `files`) are complete."""
            if finished:
                return len(files)
            if not files:
                return 0
            n = len(files) - 1        # all but the newest have a successor
            newest = files[-1]
            try:
                size = newest.stat().st_size
            except OSError:
                return n
            prev = last_size.get(newest.name)
            now = time.monotonic()
            if prev is None or prev[0] != size:
                last_size[newest.name] = (size, now)
            elif size > 0 and now - prev[1] >= STEADY_S:
                n += 1
            return n

        try:
            while True:
                if stop is not None and stop.is_set() and not cancelled:
                    cancelled = True
                    dlog("scan.winrt.cancel_requested", handed=handed)
                    try:
                        op.cancel()
                    except Exception:
                        pass
                final_status = _status(op)
                finished = final_status != _STARTED
                files = ordered_page_files(folder)
                ready = ready_count(files, finished)
                while handed < ready:
                    path = files[handed]
                    img = _decode(path)
                    if img is None:
                        first = unreadable_since.setdefault(path.name, time.monotonic())
                        if time.monotonic() - first < READ_RETRY_S:
                            break          # try this file again next poll
                        dlog("scan.winrt.unreadable", file=path.name, position=handed + 1)
                        handed += 1        # give up on it; position still counts
                        continue
                    handed += 1
                    side = "B" if (is_duplex and handed % 2 == 0) else "A"
                    dlog("scan.winrt.page", n=handed, side=side, file=path.name,
                         t=round(time.monotonic() - t0, 2))
                    try:
                        cropped, _found = billcrop.crop_item(img, margin_px=self.MARGIN_PX)
                    except Exception:
                        cropped = img   # never drop a page over a crop hiccup
                    yield ScanPage(side=side, image=cropped, raw=img,
                                   t=time.monotonic() - t0)
                if finished and handed >= len(files):
                    break
                time.sleep(POLL_S)

            if final_status == _ERROR:
                try:
                    error_code = int(op.error_code) & 0xFFFFFFFF
                except Exception:
                    error_code = 0
            dlog("scan.winrt.end", status=final_status, hresult=f"0x{error_code:08X}",
                 pages=handed, cancelled=cancelled, t=round(time.monotonic() - t0, 2))
            if final_status == _ERROR:
                raise _map_error(error_code)
        finally:
            # Make sure the scanner isn't left feeding into a folder nobody reads
            # (e.g. the consumer stopped iterating early).
            try:
                if _status(op) == _STARTED:
                    op.cancel()
                    deadline = time.monotonic() + 5
                    while _status(op) == _STARTED and time.monotonic() < deadline:
                        time.sleep(0.05)
            except Exception:
                pass
            for _ in range(20):
                shutil.rmtree(folder, ignore_errors=True)
                if not folder.exists():
                    break
                time.sleep(0.25)
            if folder.exists():
                dlog("scan.winrt.tempdir_left", folder=str(folder))
