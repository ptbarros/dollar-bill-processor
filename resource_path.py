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
        d = base / "DollarBillProcessor"
    else:
        d = Path(__file__).resolve().parent
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d
