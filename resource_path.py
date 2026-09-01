"""Resolve the app's base directory for bundled data (patterns/, models, config).

When running from source this is the repo root; when frozen by PyInstaller it is
the bundle's data dir (sys._MEIPASS). Use app_base() for any data-file lookup
instead of Path(__file__).parent so paths resolve in both cases.
"""

import sys
from pathlib import Path


def app_base() -> Path:
    if getattr(sys, "frozen", False):
        # PyInstaller onedir/onefile: bundled datas live under _MEIPASS.
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent
