"""Send to Paul: bundle a tester's report and upload it to fancydetective.com.

A report is ONE .zip, so the server can store it untouched and list it from a
small manifest. Layout:

    manifest.json              who/what/when (see build_report); format "dd-report" v1
    note.txt                   the sender's own words (also in the manifest)
    system.json                machine facts (system_info.collect())
    logs/debug_log.txt         current debug log (+ debug_log.txt.1, the previous one)
    logs/crash.log             native crash dump, if any
    settings/user_settings.yaml   their settings with API keys removed
    bills/NN_<serial>/front.jpg, back.jpg, result.json   bills they picked

Everything text-like is scrubbed: the home folder becomes "~", the account name
becomes "<user>", and anything that looks like an API key is masked.

Upload contract (implemented on the web side in dollardetective-web):
    POST {REPORT_URL}  multipart/form-data, one file field "report" (the zip)
    201 -> {"id": "R-XXXXXX", ...}     413 too large     429 rate limited     400 bad zip
No login: the upload is anonymous, protected server-side by size/rate limits.

Standalone (stdlib only; system_info is optional) so it can be tested headless.
"""

from __future__ import annotations

import getpass
import io
import json
import os
import re
import ssl
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional
import urllib.error
import urllib.request

REPORT_URL = "https://fancydetective.com/api/reports"
FORMAT = "dd-report"
FORMAT_VERSION = 1
MAX_BILLS = 10
MAX_REPORT_BYTES = 40 * 1024 * 1024        # must match the server's limit
_LOG_TAIL_BYTES = 4 * 1024 * 1024          # per log file

# Result fields worth sending per bill (keeps paths out; images go separately).
_RESULT_KEYS = (
    "serial", "position", "series_year", "front_plate", "back_plate", "fancy_types",
    "patterns", "confidence", "needs_review", "baseline_variance", "max_deviation",
    "seal_x", "seal_y", "seal_containment", "star_detected", "potential_mule",
    "front_align_angle", "front_align_flipped", "front_align_coarse",
    "back_align_coarse", "error", "note", "checked", "cropped",
)


class ReportError(Exception):
    """A report couldn't be built or sent. str() is a plain-language message."""


# --------------------------------------------------------------------------- #
# Scrubbing
# --------------------------------------------------------------------------- #
_KEY_RE = re.compile(r"\b(sk-(?:ant-)?[A-Za-z0-9_\-]{12,}|AIza[0-9A-Za-z_\-]{20,})")


def make_scrubber(home: Optional[str] = None, user: Optional[str] = None):
    """Return a function that hides the home folder, account name and API keys."""
    home = home if home is not None else str(Path.home())
    try:
        user = user if user is not None else getpass.getuser()
    except Exception:
        user = ""
    homes = {home, home.replace("\\", "/"), home.replace("/", "\\")} - {""}
    user_re = (re.compile(r"(?<![A-Za-z0-9])" + re.escape(user) + r"(?![A-Za-z0-9])", re.I)
               if user and len(user) >= 3 else None)

    def scrub(text: str) -> str:
        for h in sorted(homes, key=len, reverse=True):
            text = text.replace(h, "~")
        if user_re:
            text = user_re.sub("<user>", text)
        return _KEY_RE.sub("<api-key removed>", text)
    return scrub


def _scrub_settings(yaml_text: str, scrub) -> str:
    """Blank every *api_key* value, then scrub paths/user."""
    out = re.sub(r"(?im)^(\s*[\w.-]*api_key[\w.-]*\s*:\s*).*$", r"\1''", yaml_text)
    return scrub(out)


def _read_tail(path: Path, limit: int = _LOG_TAIL_BYTES) -> str:
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - limit))
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    if size > limit:
        text = f"[... first {size - limit:,} bytes omitted ...]\n" + text
    return text


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s or "")[:24] or "bill"


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_report(dest: Path, *, sender: str, note: str,
                 log_paths: Iterable[Path] = (), crash_path: Optional[Path] = None,
                 settings_path: Optional[Path] = None, bills: Iterable[dict] = (),
                 system: Optional[dict] = None, app_version: str = "",
                 scrub=None) -> dict:
    """Write the report zip to `dest` and return its manifest.

    `bills` are result dicts (front_file/back_file + fields); at most MAX_BILLS
    are included. Missing files are skipped, never fatal. Raises ReportError if
    the finished zip is over MAX_REPORT_BYTES.
    """
    scrub = scrub or make_scrubber()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    bills = list(bills)[:MAX_BILLS]
    if system is None:
        try:
            from dollar_detective.system_info import collect
            system = collect()
        except Exception:
            system = {}

    manifest = {
        "format": FORMAT, "version": FORMAT_VERSION,
        "report_id": uuid.uuid4().hex[:12],
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sender": (sender or "").strip()[:80],
        "note": (note or "").strip()[:4000],
        "app_version": app_version or system.get("app_version", ""),
        "edition": system.get("edition", ""),
        "os": system.get("os", ""),
        "contents": {"logs": [], "crash_log": False, "settings": False,
                     "system": bool(system), "bills": []},
    }

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("note.txt", manifest["note"] + "\n")
        z.writestr("system.json", scrub(json.dumps(system, indent=2)))
        for p in log_paths:
            p = Path(p)
            if p.is_file():
                z.writestr(f"logs/{p.name}", scrub(_read_tail(p)))
                manifest["contents"]["logs"].append(p.name)
        if crash_path and Path(crash_path).is_file() and Path(crash_path).stat().st_size:
            z.writestr("logs/crash.log", scrub(_read_tail(Path(crash_path))))
            manifest["contents"]["crash_log"] = True
        if settings_path and Path(settings_path).is_file():
            text = Path(settings_path).read_text(encoding="utf-8", errors="replace")
            z.writestr("settings/user_settings.yaml", _scrub_settings(text, scrub))
            manifest["contents"]["settings"] = True
        for i, r in enumerate(bills, 1):
            serial = str(r.get("serial") or "")
            folder = f"bills/{i:02d}_{_safe_name(serial)}"
            row = {k: r.get(k) for k in _RESULT_KEYS if k in r}
            files = []
            for side in ("front", "back"):
                src = r.get(f"{side}_file")
                if src and Path(src).is_file():
                    ext = Path(src).suffix.lower() or ".jpg"
                    # Images are already compressed: store, don't deflate again.
                    z.write(src, f"{folder}/{side}{ext}", compress_type=zipfile.ZIP_STORED)
                    files.append(f"{side}{ext}")
            row["included_images"] = files
            z.writestr(f"{folder}/result.json", scrub(json.dumps(row, indent=2, default=str)))
            manifest["contents"]["bills"].append({"serial": serial, "images": files})
        z.writestr("manifest.json", json.dumps(manifest, indent=2))

    size = dest.stat().st_size
    manifest["size_bytes"] = size
    if size > MAX_REPORT_BYTES:
        raise ReportError(
            f"The report is {size / 1e6:.0f} MB, over the {MAX_REPORT_BYTES / 1e6:.0f} MB limit. "
            "Include fewer bills and try again.")
    return manifest


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #
def _multipart(zip_path: Path):
    boundary = "----ddreport" + uuid.uuid4().hex
    head = (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="report"; filename="{zip_path.name}"\r\n'
            "Content-Type: application/zip\r\n\r\n").encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    body = head + zip_path.read_bytes() + tail
    return body, f"multipart/form-data; boundary={boundary}"


def _open(req, timeout):
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.URLError as e:
        # Fresh Windows installs can lack a root certificate (see the CUDA SSL
        # issue); retry once with certifi's bundle if it's available.
        if isinstance(getattr(e, "reason", None), ssl.SSLError):
            try:
                import certifi
                ctx = ssl.create_default_context(cafile=certifi.where())
                return urllib.request.urlopen(req, timeout=timeout, context=ctx)
            except ImportError:
                pass
        raise


def upload_report(zip_path: Path, url: str = REPORT_URL, timeout: int = 120,
                  app_version: str = "") -> str:
    """Upload the zip; return the server's reference code (e.g. "R-7K3QX2").
    Raises ReportError with a plain-language message on any failure."""
    zip_path = Path(zip_path)
    body, ctype = _multipart(zip_path)
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": ctype,
        "User-Agent": f"DollarDetective/{app_version or 'unknown'} (send-report)",
        "Accept": "application/json"})
    try:
        with _open(req, timeout) as resp:
            data = json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        if e.code == 413:
            raise ReportError("The report is too large for the server. Include fewer bills.")
        if e.code == 429:
            raise ReportError("Too many reports sent recently. Please wait a while and try again.")
        if e.code == 404:
            raise ReportError("The report service isn't available yet (404).")
        raise ReportError(f"The server refused the report (HTTP {e.code}).")
    except urllib.error.URLError as e:
        raise ReportError(f"Couldn't reach fancydetective.com. Check the internet connection.\n({e.reason})")
    except (TimeoutError, OSError) as e:
        raise ReportError(f"Sending timed out or failed: {e}")
    except ValueError:
        raise ReportError("The server's reply wasn't understood.")
    ref = str(data.get("id") or "").strip()
    if not ref:
        raise ReportError("The server didn't return a reference code.")
    return ref
