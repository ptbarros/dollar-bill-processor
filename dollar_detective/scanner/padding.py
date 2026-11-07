"""Trim the white padding a scanner driver adds after the paper ends.

WIA/WinRT drivers stop scanning at the paper's end but still return a page as
tall as the requested window, filled with white. billcrop (vendored verbatim
from fastfoto640) only trims rows that are white in EVERY pixel -- but the
FF-640's front (A) pages can end with a short grey dash on the very last row
(~18 px of RGB 109 seen on a real page), so nothing got trimmed, the white
block was taken for part of the bill, and fronts came out with a white slab
under them (Paul's strap 017, 2026-11-07). Here a row counts as padding when
nearly all of its middle is near-white, so stray marks and JPEG ringing don't
stop the trim.
"""

import numpy as np

NEAR_WHITE = 235      # min channel value counted as white (bill paper is ~205)
ROW_WHITE_FRAC = 0.95   # a stray dash of up to 5% of the row is still padding


def trim_padding(img: np.ndarray) -> np.ndarray:
    """Drop trailing padding rows; returns the image unchanged if there are none."""
    h, w = img.shape[:2]
    band = max(4, int(w * 0.05))          # well clear of any dark edge columns
    core = img[:, band:w - band] if w > 4 * band else img
    if core.ndim == 3:
        core = core.min(axis=2)
    white_frac = (core >= NEAR_WHITE).mean(axis=1)
    rows = np.flatnonzero(white_frac < ROW_WHITE_FRAC)
    if rows.size == 0:
        return img
    return img[: rows[-1] + 1]
