"""Windows TWAIN scanning through a small helper program (dd_twain_helper.exe).

Why: on the Epson FF-640, Epson's own TWAIN driver with its border detection +
deskew turned on gives tight, square crops with all four corners intact and
runs at ~0.64 s/bill with no buffered tail -- better than the WIA/WinRT path
(~0.9 s/bill, trailing corner cut at paper end). Measured on Paul's laptop,
2026-11-07, Epson Scan 2 6.4.8 and 6.7.81.

Why a helper: Epson only installs a 32-bit TWAIN data source and the app is
64-bit, and TWAIN data sources load in-process. The helper (tools/twain_helper,
C#, built for x86) talks TWAIN and drops finished pages into a folder, printing
one JSON line per event; this module drives it and hands pages on.

Helper protocol (stdout, one JSON object per line):
  list  -> {"event":"source","name":..,"manufacturer":..,"family":..,"protocol":..}
  caps  -> {"event":"caps","source":..,"caps":{..}}
  scan  -> {"event":"config",..} {"event":"waiting"} {"event":"page","n":..,"side":"A|B",
           "file":..,"t":..} {"event":"batch_end","pages":..}
           {"event":"error","rc":..,"cc":..,"msg":..} {"event":"end"}
  Stop a scan by creating STOP (finish the stack that's feeding, then stop) or
  STOPNOW (at once) in its --out folder.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import cv2
import numpy as np

from .backend import ScanError, ScannerBackend, ScanPage, ScannerDevice, stop_now

try:
    from dollar_detective.debug_logger import dlog
except Exception:   # pragma: no cover
    def dlog(*_a, **_k):
        pass

HELPER_NAME = "dd_twain_helper.exe"
STOP_FILE = "STOP"          # finish the feeding stack, then stop
STOP_NOW_FILE = "STOPNOW"   # stop at once
LIST_TIMEOUT_S = 20
STOP_QUIET_S = 15           # graceful stop escalates if the scanner goes quiet this long
STOP_GRACE_S = 20


def helper_path() -> Optional[Path]:
    """The bundled helper, a dev build, or $DD_TWAIN_HELPER; None if absent."""
    cands = []
    env = os.environ.get("DD_TWAIN_HELPER")
    if env:
        cands.append(Path(env))
    try:
        from dollar_detective.resource_path import app_base
        base = app_base()
        cands += [base / "scanner_helpers" / HELPER_NAME,
                  base / "tools" / "twain_helper" / "bin" / HELPER_NAME]
    except Exception:
        pass
    for c in cands:
        if c.is_file():
            return c
    return None


def _norm(name: str) -> str:
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def match_source(device_name: str, sources: List[dict]) -> Optional[dict]:
    """The TWAIN source for a scanner WIA/WinRT reports by name, e.g.
    'Epson FF-640' <-> 'EPSON FF-640', 'CANON R40 USB #3' <-> 'CANON R40'."""
    dev = _norm(device_name).replace("usb", "")
    best = None
    for s in sources:
        src = _norm(s.get("name", ""))
        if not src:
            continue
        if src in dev or dev.startswith(src) or src.startswith(dev):
            if best is None or len(src) > len(_norm(best.get("name", ""))):
                best = s
    return best


def _creationflags():
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _run_helper(args: List[str], timeout: float) -> List[dict]:
    exe = helper_path()
    if exe is None:
        raise ScanError("TWAIN helper not found.", cause="NOHELPER")
    try:
        out = subprocess.run([str(exe)] + args, capture_output=True, text=True,
                             timeout=timeout, creationflags=_creationflags())
    except subprocess.TimeoutExpired:
        raise ScanError("The TWAIN driver didn't respond.", cause="TIMEOUT")
    events = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    return events


class _EpsonSettingsGuard:
    """Epson's TWAIN driver rewrites its saved settings (UserSettings.SF2) when
    it closes -- which would change the user's own Epson Scan / FastFoto
    setup. Snapshot that folder before the helper runs and put it back after."""

    def __init__(self):
        appdata = os.environ.get("APPDATA")
        self.root = Path(appdata) / "Epson" / "Epson Scan 2" / "Settings" if appdata else None
        self.saved: Dict[Path, bytes] = {}

    def __enter__(self):
        if self.root and self.root.is_dir():
            for p in self.root.rglob("*"):
                if p.is_file():
                    try:
                        self.saved[p] = p.read_bytes()
                    except OSError:
                        pass
        return self

    def __exit__(self, *exc):
        if not self.root or not self.root.is_dir():
            return
        try:
            for p in self.root.rglob("*"):
                if p.is_file() and p not in self.saved:
                    p.unlink()           # created by the driver this run
            for p, data in self.saved.items():
                try:
                    if not p.exists() or p.read_bytes() != data:
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes(data)
                except OSError:
                    pass
        except Exception as e:
            dlog("scan.twain.settings_restore_failed", error=repr(e))


class TwainBackend(ScannerBackend):
    """TWAIN via dd_twain_helper.exe. Lists installed TWAIN sources (installed
    != connected, so the Windows backend pairs these with connected WIA
    devices by name)."""

    label = "Windows scanner (TWAIN)"
    can_detect_paper = False   # the helper waits for paper itself, across stacks
    MARGIN_PX = 6
    in_flight_lost = False     # set if a Stop couldn't let in-flight bills finish

    def __init__(self):
        if helper_path() is None:
            raise RuntimeError("TWAIN helper not bundled")

    def sources(self) -> List[dict]:
        try:
            return [e for e in _run_helper(["list"], LIST_TIMEOUT_S)
                    if e.get("event") == "source"]
        except Exception as e:
            dlog("scan.twain.list_failed", error=repr(e))
            return []

    def list_devices(self) -> List[ScannerDevice]:
        return [ScannerDevice(id="twain:" + s["name"], name=s["name"]) for s in self.sources()]

    def describe(self, device_id: Optional[str] = None) -> dict:
        info = {"backend": self.label, "sources": self.sources()}
        name = (device_id or "").split("twain:", 1)[-1]
        if name:
            caps = [e for e in _run_helper(["caps", "--source", name], 60)
                    if e.get("event") == "caps"]
            info["caps"] = caps[0].get("caps") if caps else None
        return info

    def scan_pages(self, device_id: Optional[str] = None, *,
                   dpi: int = 300, duplex: bool = True, color: bool = True,
                   stop=None, **options) -> Iterator[ScanPage]:
        exe = helper_path()
        if exe is None:
            raise ScanError("TWAIN helper not found.", cause="NOHELPER")
        name = (device_id or "").split("twain:", 1)[-1]
        if not name:
            srcs = self.sources()
            if not srcs:
                raise ScanError("No TWAIN scanner found.", cause="OFFLINE")
            name = srcs[0]["name"]
        folder = Path(tempfile.mkdtemp(prefix="dd_twain_"))
        args = [str(exe), "scan", "--source", name, "--out", str(folder),
                "--dpi", str(int(dpi)), "--duplex", "1" if duplex else "0",
                "--color", "1" if color else "0", "--crop", "1", "--deskew", "1",
                "--once", "1" if options.get("once") else "0"]
        dlog("scan.twain.start", source=name, folder=str(folder), dpi=dpi, duplex=duplex)
        with _EpsonSettingsGuard():
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, creationflags=_creationflags())
            yield from self._pump(proc, folder, stop)

    def _pump(self, proc, folder: Path, stop) -> Iterator[ScanPage]:
        from . import billcrop
        from .padding import trim_padding

        events: "queue.Queue" = queue.Queue()

        def reader():
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("{"):
                    try:
                        events.put(json.loads(line))
                    except ValueError:
                        pass
            events.put({"event": "_eof"})

        threading.Thread(target=reader, daemon=True).start()
        driver_crops = True
        pages = 0
        error = None
        empty = False
        stop_sent = now_sent = None       # times STOP / STOPNOW were written
        last_event = time.monotonic()
        t0 = time.monotonic()

        def drop(name):
            try:
                (folder / name).write_text("stop")
            except OSError:
                pass

        try:
            while True:
                now_t = time.monotonic()
                if stop is not None and stop.is_set() and stop_sent is None:
                    stop_sent = last_event = now_t
                    drop(STOP_FILE)        # finish the stack that's feeding
                    dlog("scan.twain.stop_requested", pages=pages)
                if now_sent is None and (stop_now(stop) or (
                        stop_sent and now_t - last_event > STOP_QUIET_S)):
                    now_sent = now_t       # app closing, or the scanner went quiet
                    drop(STOP_NOW_FILE)
                    dlog("scan.twain.stop_now", pages=pages)
                if now_sent and now_t - now_sent > STOP_GRACE_S:
                    dlog("scan.twain.stop_timeout")
                    proc.kill()
                    break
                try:
                    ev = events.get(timeout=0.2)
                except queue.Empty:
                    continue
                last_event = time.monotonic()
                kind = ev.get("event")
                if kind == "config":
                    driver_crops = bool(ev.get("border_detect", True))
                    dlog("scan.twain.config", **{k: v for k, v in ev.items() if k != "event"})
                elif kind == "page":
                    path = Path(ev.get("file", ""))
                    img = None
                    try:
                        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8),
                                           cv2.IMREAD_COLOR)
                    except Exception:
                        pass
                    if img is None:
                        dlog("scan.twain.unreadable", file=str(path))
                        continue
                    pages += 1
                    raw = img
                    if not driver_crops:
                        try:
                            img, _ = billcrop.crop_item(trim_padding(img), margin_px=self.MARGIN_PX)
                        except Exception:
                            pass
                    try:
                        path.unlink()
                    except OSError:
                        pass
                    side = "B" if ev.get("side") == "B" else "A"
                    dlog("scan.twain.page", n=pages, side=side, t=ev.get("t"))
                    yield ScanPage(side=side, image=img, raw=raw,
                                   t=float(ev.get("t", time.monotonic() - t0)))
                elif kind == "empty":
                    empty = True
                    dlog("scan.twain.empty", t=ev.get("t"))
                elif kind == "warning":
                    dlog("scan.twain.warning", **{k: v for k, v in ev.items() if k != "event"})
                    if ev.get("in_flight_lost"):
                        self.in_flight_lost = True
                elif kind in ("waiting", "batch_end", "stopping"):
                    dlog(f"scan.twain.{kind}", **{k: v for k, v in ev.items() if k != "event"})
                elif kind == "error":
                    error = ev
                    dlog("scan.twain.error", **{k: v for k, v in ev.items() if k != "event"})
                elif kind in ("end", "_eof"):
                    break
        finally:
            if proc.poll() is None:
                drop(STOP_NOW_FILE)        # consumer gave up: stop at once
                try:
                    proc.wait(timeout=STOP_GRACE_S)
                except subprocess.TimeoutExpired:
                    proc.kill()
            shutil.rmtree(folder, ignore_errors=True)
            dlog("scan.twain.end", pages=pages, rc=proc.returncode,
                 t=round(time.monotonic() - t0, 2))
        if empty and pages == 0 and error is None:
            raise ScanError("No paper in the feeder.", cause="PE")
        if error is not None and not (stop is not None and stop.is_set()):
            msg = str(error.get("msg", "error"))
            # Couldn't get the scanner going at all (no TWAIN manager, source
            # missing, open/enable refused) vs. a problem during a scan (jam).
            # Only the former may fall back to WIA.
            startup = pages == 0 and any(k in msg for k in (
                "OPENDS", "ENABLEDS", "source not found", "data source manager",
                "DSM_Entry", "MSG_OPENDSM"))
            raise ScanError(f"TWAIN: {msg} (rc={error.get('rc')} cc={error.get('cc')})",
                            cause="TWAIN_START" if startup else "TWAIN")


class WindowsScannerBackend(ScannerBackend):
    """Windows: one backend that picks TWAIN or WIA (WinRT) per the setting.

    "auto": TWAIN when the connected scanner has a matching TWAIN source; WIA
    only if TWAIN can't START (no helper/source, or the driver won't open).
    Normal TWAIN outcomes -- empty feeder, jam, stop -- never fall back: on
    Paul's laptop a "no paper" fallback sent the Scanner Test to WIA, whose
    Epson driver (1.1.0.1) then crashed the Windows scanner service.
    Not sticky: every scan tries TWAIN first. "twain"/"wia": force one."""

    label = "Windows scanner"

    def __init__(self):
        self.wia = None
        self.twain = None
        try:
            from .winrt_backend import WinRtScannerBackend
            self.wia = WinRtScannerBackend()
        except Exception:
            pass
        try:
            self.twain = TwainBackend()
        except Exception:
            pass
        if self.wia is None and self.twain is None:
            raise RuntimeError("no Windows scanner API available")
        self._twain_failed = False
        self.last_driver = None

    @property
    def can_detect_paper(self):
        return False

    def list_devices(self) -> List[ScannerDevice]:
        if self.wia is not None:
            return self.wia.list_devices()      # connected scanners only
        return self.twain.list_devices() if self.twain else []

    def _twain_source_for(self, device_id: Optional[str]) -> Optional[str]:
        if self.twain is None:
            return None
        if device_id and device_id.startswith("twain:"):
            return device_id
        devs = self.list_devices()
        dev = next((d for d in devs if d.id == device_id), devs[0] if devs else None)
        if dev is None:
            return None
        src = match_source(dev.name, self.twain.sources())
        return "twain:" + src["name"] if src else None

    def choose(self, device_id: Optional[str], driver: str = "auto") -> tuple:
        """(backend, device_id) to scan with."""
        if driver != "wia":
            src = self._twain_source_for(device_id)
            if src:
                return self.twain, src
            if driver == "twain":
                raise ScanError("No TWAIN driver found for this scanner.", cause="NOTWAIN")
        if self.wia is None:
            raise ScanError("WIA scanning isn't available.", cause="NOWIA")
        return self.wia, device_id

    def describe(self, device_id: Optional[str] = None) -> dict:
        info = {"backend": self.label}
        if self.wia is not None:
            try:
                info["wia"] = self.wia.describe(device_id)
            except Exception as e:
                info["wia"] = {"error": repr(e)}
        if self.twain is not None:
            try:
                src = self._twain_source_for(device_id)
                info["twain"] = self.twain.describe(src) if src else {
                    "sources": self.twain.sources(), "matched": None}
            except Exception as e:
                info["twain"] = {"error": repr(e)}
        else:
            info["twain"] = {"error": "helper not available"}
        info["name"] = (info.get("wia") or {}).get("name")
        return info

    def scan_pages(self, device_id: Optional[str] = None, *,
                   dpi: int = 300, duplex: bool = True, color: bool = True,
                   stop=None, driver: str = "auto", **options) -> Iterator[ScanPage]:
        backend, dev = self.choose(device_id, driver)
        self.last_driver = "twain" if backend is self.twain else "wia"
        dlog("scan.windows.driver", driver=self.last_driver, requested=driver, device=dev)
        got = 0
        try:
            for page in backend.scan_pages(dev, dpi=dpi, duplex=duplex, color=color,
                                           stop=stop, **options):
                got += 1
                yield page
        except ScanError as e:
            if (backend is self.twain and driver == "auto" and got == 0 and self.wia
                    and e.cause in ("TWAIN_START", "NOHELPER")):
                self._twain_failed = True       # for the log/report only
                dlog("scan.windows.twain_fallback", error=str(e))
                self.last_driver = "wia"
                yield from self.wia.scan_pages(device_id, dpi=dpi, duplex=duplex,
                                               color=color, stop=stop, **options)
                return
            raise
