"""In-app update check against the GitHub Releases API.

Compares the running ``version.py`` __version__ to the latest *stable* (non-
prerelease) GitHub release and, when a newer one exists, helps the user get the
right download for their edition. Pure stdlib (urllib + json) so it adds no
dependency and runs inside the frozen build.

Design:
  * check_for_update() -> UpdateInfo | None      (network, never raises)
  * detect_edition()   -> which build is running (picks the matching asset)
  * download_asset()   -> fetch an installer/AppImage to a temp file
  * launch_installer_and_exit() (Windows) / the caller opens the page elsewhere

Prereleases (hyphenated tags like v1.4.3-rc1) are ignored so test builds never
prompt real users to "update".
"""

import json
import os
import re
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from version import __version__

REPO = "ptbarros/dollar-bill-processor"
RELEASES_API = f"https://api.github.com/repos/{REPO}/releases?per_page=30"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"


def _parse_version(tag: Optional[str]):
    """'v1.4.5' / '1.4.5' -> (1, 4, 5); a prerelease (has a hyphen) -> None."""
    if not tag:
        return None
    t = str(tag).lstrip("vV").strip()
    if "-" in t:                      # prerelease, e.g. 1.4.0-ci2 / 1.4.3-rc1
        return None
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)$", t)
    return tuple(int(x) for x in m.groups()) if m else None


@dataclass
class UpdateInfo:
    latest_version: str               # "1.4.6"
    current_version: str              # __version__
    release_url: str                  # release html_url (browser fallback)
    assets: List[dict] = field(default_factory=list)  # [{name, url}]
    notes: str = ""                   # release body (markdown "What's Changed")

    def asset_for_current_edition(self) -> Optional[dict]:
        return _asset_for_edition(self.assets, detect_edition())


def detect_edition() -> str:
    """Which build is running -> 'cuda' | 'directml' | 'openvino' | 'appimage'
    | 'macos' | 'source'. Drives which release asset to download."""
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "appimage"
    # Windows editions:
    if os.environ.get("DBP_FORCE_TORCH") == "1" or "torch" in sys.modules:
        return "cuda"
    try:
        import onnxruntime as ort
        provs = ort.get_available_providers()
        if "OpenVINOExecutionProvider" in provs:
            return "openvino"
        if "DmlExecutionProvider" in provs:
            return "directml"
    except Exception:
        pass
    return "openvino"                 # default Windows edition


def _asset_for_edition(assets: List[dict], edition: str) -> Optional[dict]:
    """Pick the download asset matching the running edition, or None."""
    def find(pred):
        for a in assets:
            if pred(a["name"]):
                return a
        return None

    if edition == "cuda":
        return find(lambda n: n.endswith("-cuda-setup.exe"))
    if edition == "directml":
        return find(lambda n: n.endswith("-directml-setup.exe"))
    if edition == "openvino":
        # the default installer: *-setup.exe with no edition tag
        return find(lambda n: n.endswith("-setup.exe")
                    and "-cuda-" not in n and "-directml-" not in n)
    if edition == "appimage":
        return find(lambda n: n.endswith(".AppImage"))
    if edition == "macos":
        return find(lambda n: n.endswith(".dmg"))
    return None


def check_for_update(timeout: int = 6) -> Optional[UpdateInfo]:
    """Return UpdateInfo if a newer STABLE release exists, else None. Never raises
    (returns None on any network/parse error or when already up to date)."""
    current = _parse_version(__version__)
    if current is None:               # dev build with an odd version -> don't nag
        return None
    try:
        req = urllib.request.Request(
            RELEASES_API,
            headers={"Accept": "application/vnd.github+json",
                     "User-Agent": "DollarDetective-Updater"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            releases = json.load(resp)
    except Exception:
        return None

    best_ver = current
    best_rel = None
    for rel in releases:
        if rel.get("prerelease") or rel.get("draft"):
            continue
        v = _parse_version(rel.get("tag_name"))
        if v is not None and v > best_ver:
            best_ver, best_rel = v, rel
    if best_rel is None:
        return None

    assets = [{"name": a["name"], "url": a["browser_download_url"]}
              for a in best_rel.get("assets", [])]
    return UpdateInfo(
        latest_version=".".join(map(str, best_ver)),
        current_version=__version__,
        release_url=best_rel.get("html_url", RELEASES_PAGE),
        assets=assets,
        notes=(best_rel.get("body") or "").strip(),
    )


def download_asset(url: str, progress_cb: Optional[Callable[[int, int], None]] = None,
                   timeout: int = 30) -> Optional[str]:
    """Download an asset to a temp file; return its path or None on failure.
    progress_cb(downloaded_bytes, total_bytes) is called during the download."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DollarDetective-Updater"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            suffix = os.path.splitext(url.split("?")[0])[1] or ".bin"
            fd, path = tempfile.mkstemp(suffix=suffix, prefix="DollarDetective-update-")
            done = 0
            with os.fdopen(fd, "wb") as out:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if progress_cb:
                        progress_cb(done, total)
        return path
    except Exception:
        return None


def launch_installer_and_exit(installer_path: str) -> bool:
    """Windows: launch the downloaded setup.exe (the per-user Inno installer) so
    the caller can immediately exit and let it replace the app. Returns True if
    the installer was launched. On non-Windows this is a no-op returning False
    (the caller should open the release page instead)."""
    if sys.platform != "win32":
        return False
    try:
        os.startfile(installer_path)  # noqa: has to run on Windows  # type: ignore[attr-defined]
        return True
    except Exception:
        return False
