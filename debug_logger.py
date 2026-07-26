"""
Lightweight file-based debug logger for diagnosing the intermittent
"lost review state / auto-archive" issue.

Why this exists
---------------
The app previously only used print() to stdout, which is invisible when the
GUI is launched from a shortcut / packaged build (the FIL's setup). This module
writes a rotating, timestamped log to a known-writable location so we can see
EXACTLY what happened leading up to a suspected reset/archive.

Usage
-----
    from debug_logger import dlog, fingerprint, get_log_path

    dlog("crop.requested", count=len(results))
    dlog("state.before", **fingerprint(self.current_results))

`fingerprint()` summarizes the per-bill review flags so a silent reset shows up
as the counts collapsing to zero between two log lines.

The log file lives next to .session_recovery.json (the project dir), named
debug_log.txt, and rotates at ~2 MB x 5 files.
"""

import logging
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGER_NAME = "dollarbill.debug"
_logger = None
_log_path = None
_lock = threading.Lock()


def _log_dir() -> Path:
    """Directory for the log file: the project root (same place as the
    recovery file). Falls back to the user's home dir if that is not
    writable for any reason."""
    candidate = Path(__file__).resolve().parent
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        test = candidate / ".dbglog_write_test"
        test.write_text("ok")
        test.unlink()
        return candidate
    except Exception:
        home = Path.home()
        return home


def _init_logger() -> logging.Logger:
    global _logger, _log_path
    if _logger is not None:
        return _logger
    with _lock:
        if _logger is not None:
            return _logger
        logger = logging.getLogger(_LOGGER_NAME)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        _log_path = _log_dir() / "debug_log.txt"
        try:
            handler = RotatingFileHandler(
                str(_log_path), maxBytes=2 * 1024 * 1024, backupCount=5,
                encoding="utf-8"
            )
        except Exception:
            # Last resort: stream handler so we never crash the app over logging
            handler = logging.StreamHandler()

        fmt = logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(threadName)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

        # Also mirror to stdout for interactive/dev runs.
        try:
            stream = logging.StreamHandler()
            stream.setFormatter(fmt)
            logger.addHandler(stream)
        except Exception:
            pass

        _logger = logger
        logger.info("=" * 70)
        logger.info("debug_logger initialized -> %s", _log_path)
        logger.info("=" * 70)
        return _logger


def get_log_path() -> str:
    """Absolute path to the current log file (for surfacing in the UI)."""
    _init_logger()
    return str(_log_path) if _log_path else ""


def dlog(event: str, **context) -> None:
    """Log a structured event. Never raises — logging must not break the app.

    Example line:
        2026-07-26 14:03:11.512 [MainThread] crop.requested | count=12 selected_index=59
    """
    try:
        logger = _init_logger()
        if context:
            ctx = " ".join(f"{k}={_short(v)}" for k, v in context.items())
            logger.info("%s | %s", event, ctx)
        else:
            logger.info("%s", event)
    except Exception:
        pass


def dlog_exc(event: str, **context) -> None:
    """Like dlog() but also records the current exception traceback."""
    try:
        logger = _init_logger()
        ctx = " ".join(f"{k}={_short(v)}" for k, v in context.items())
        logger.exception("%s | %s", event, ctx)
    except Exception:
        pass


def fingerprint(results) -> dict:
    """Summarize the per-bill review state so a silent reset is visible as the
    counts dropping between two consecutive log lines.

    Returns a dict suitable for **-splatting into dlog().
    """
    try:
        results = results or []
        total = len(results)
        viewed = sum(1 for r in results if _truthy(r.get("viewed")))
        checked = sum(1 for r in results if _truthy(r.get("checked")))
        cropped = sum(1 for r in results if _truthy(r.get("cropped")))
        sent = sum(1 for r in results if _truthy(r.get("sent_for_review")))
        labeled = sum(1 for r in results if _has_label(r))
        return {
            "n": total,
            "viewed": viewed,
            "checked": checked,
            "cropped": cropped,
            "sent": sent,
            "labeled": labeled,
        }
    except Exception:
        return {"n": "?", "viewed": "?", "checked": "?", "cropped": "?"}


def _has_label(r) -> bool:
    for key in ("note", "custom_label", "label", "pattern_override"):
        v = r.get(key)
        if v not in (None, "", False):
            return True
    return False


def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


def _short(v, limit: int = 120) -> str:
    try:
        s = str(v)
    except Exception:
        s = "<unprintable>"
    if len(s) > limit:
        return s[:limit] + "…"
    return s
