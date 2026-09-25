"""Resolve the app's base directory for bundled data (patterns/, models, config).

When running from source this is the repo root; when frozen by PyInstaller it is
the bundle's data dir (sys._MEIPASS). Use app_base() for any data-file lookup
instead of Path(__file__).parent so paths resolve in both cases.
"""

import os
import sys
from pathlib import Path


def app_base() -> Path:
    """Base dir for bundled READ-ONLY data (patterns/, model, config)."""
    if getattr(sys, "frozen", False):
        # PyInstaller onedir/onefile: bundled datas live under _MEIPASS.
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def user_data_dir() -> Path:
    """Writable per-user dir for user data (settings, corrections, session
    recovery, saved patterns).

    A frozen bundle is read-only, so these must NOT live next to the code. When
    frozen, use the platform's per-user config dir; from source, keep the repo
    root so dev behavior is unchanged.
    """
    if getattr(sys, "frozen", False):
        if sys.platform == "win32":
            base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
        elif sys.platform == "darwin":
            base = Path.home() / "Library" / "Application Support"
        else:
            base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
        d = base / "DollarDetective"
        # App was renamed DollarBillProcessor -> Dollar Detective; move an
        # existing user-data dir across so upgraders keep their settings.
        _migrate_legacy_user_data(base / "DollarBillProcessor", d)
    else:
        d = Path(__file__).resolve().parent
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


def content_dir() -> Path:
    """Visible per-user folder for scan batches (the "Straps" tree used by Monitor
    mode). Defaults to ~/DollarDetective, which is OneDrive-safe on Windows --
    Path.home() is the real user profile (C:\\Users\\<user>), not a OneDrive-
    redirected Documents/Desktop. A user setting (processing.data_folder) can point
    it anywhere, including the Desktop if they insist. Config and user patterns do
    NOT live here -- they stay in user_data_dir().
    """
    override = ""
    try:  # lazy import: settings_manager imports this module
        from settings_manager import get_settings
        override = (get_settings().processing.data_folder or "").strip()
    except Exception:
        override = ""
    d = Path(override).expanduser() if override else (Path.home() / "DollarDetective")
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


def straps_dir() -> Path:
    """Folder holding the per-batch ("strap") folders, under content_dir()."""
    d = content_dir() / "Straps"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


def is_in_onedrive(path) -> bool:
    """True if `path` looks like it lives inside a OneDrive-synced tree (used for a
    soft warning, never to block the user's choice)."""
    try:
        p = str(Path(path).expanduser()).replace("\\", "/").lower()
        return "/onedrive" in p
    except Exception:
        return False


def _migrate_legacy_user_data(old: Path, new: Path) -> None:
    """One-time move of the pre-rename user-data dir (DollarBillProcessor ->
    DollarDetective) so upgrading users keep their settings, corrections, saved
    patterns and session recovery. Runs only when the old dir exists and the new
    one does not yet; any failure is swallowed so it can never block startup.
    """
    try:
        if new.exists() or not old.is_dir():
            return
        import shutil
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(new))
    except Exception:
        pass
