"""Names for Monitor-mode batch ("strap") folders.

The user picks the format (processing.batch_name_format):
  * "number"       -> 001            (FIL: plain sequential number)
  * "date"         -> 2026-09-25     (a date; a same-day repeat gets " (2)", ...)
  * "number_date"  -> 001 - 2026-09-25
  * "date_number"  -> 2026-09-25 - 001

The sequential number comes from a monotonic counter in settings
(processing.batch_counter) so it never reuses a number even if a folder is
deleted or renamed. Actual folder creation de-dupes against what's on disk, so
two batches never collide regardless of format.
"""

import datetime
from pathlib import Path

PAD = 3


def _today() -> str:
    return datetime.date.today().isoformat()


def format_batch_name(fmt: str, number: int, today: str = None) -> str:
    """Render a batch name for `fmt` and sequential `number`. Pure/testable."""
    today = today or _today()
    n = f"{number:0{PAD}d}"
    if fmt == "date":
        return today
    if fmt == "number_date":
        return f"{n} - {today}"
    if fmt == "date_number":
        return f"{today} - {n}"
    return n  # "number" (default)


def next_batch_number(settings) -> int:
    """Advance and persist the monotonic batch counter; return the new number."""
    try:
        n = int(getattr(settings.processing, "batch_counter", 0) or 0) + 1
        settings.processing.batch_counter = n
        settings.save()
        return n
    except Exception:
        # Never block a batch over the counter; fall back to a time-based number.
        return int(datetime.datetime.now().strftime("%H%M%S"))


def make_batch_dir(straps_dir: Path, fmt: str, settings) -> Path:
    """Create and return a unique batch folder under `straps_dir` for `fmt`.

    De-dupes on disk: if the chosen name already exists (common for the "date"
    format run twice in a day), append " (2)", " (3)", ... until free.
    """
    number = next_batch_number(settings)
    base = format_batch_name(fmt, number)
    straps_dir = Path(straps_dir)
    straps_dir.mkdir(parents=True, exist_ok=True)
    name, i = base, 2
    while (straps_dir / name).exists():
        name = f"{base} ({i})"
        i += 1
    d = straps_dir / name
    d.mkdir(parents=True, exist_ok=True)
    return d
