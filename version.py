"""Application version.

``__version__`` is the human-facing semantic version. Bump it by hand when you
ship a change worth distinguishing (this is what the FIL's zip install shows in
Help > About).

``get_version_string()`` additionally appends the live git short hash *when run
from a git checkout* (your dev machine), e.g. "1.3.0 (a1b2c3d)". The hash is read
at runtime, so there is no chicken-and-egg with committing it. Zip installs have
no .git folder, so they simply show the plain version.
"""

import subprocess
from pathlib import Path

__version__ = "1.4.1"


def _git_short_hash():
    """Return the short commit hash (with a -dirty suffix if the tree has
    uncommitted changes) when running from a git checkout, else None."""
    root = Path(__file__).resolve().parent
    if not (root / ".git").exists():
        return None
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root), capture_output=True, text=True, timeout=2,
        )
        if rev.returncode != 0 or not rev.stdout.strip():
            return None
        short = rev.stdout.strip()

        # --untracked-files=no: the repo carries many untracked scratch files
        # (archive/, test_bills/, chat logs, etc.), so only uncommitted changes
        # to TRACKED files should count as "dirty".
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(root), capture_output=True, text=True, timeout=2,
        )
        if status.returncode == 0 and status.stdout.strip():
            short += "-dirty"
        return short
    except Exception:
        return None


def get_version_string():
    """Human-facing version string, e.g. '1.3.0' or '1.3.0 (a1b2c3d)'."""
    short = _git_short_hash()
    return f"{__version__} ({short})" if short else __version__
