"""Cut off the white padding a scanner driver adds after the paper ends.

WIA/WinRT drivers stop scanning at the paper's end but still return a page as
tall as the requested window, filled with white -- and the FF-640's front (A)
pages END with junk below that white: a full-width grey band ~5 rows tall on
uncompressed (DIB) pages, a short grey dash on others. billcrop (vendored
verbatim from fastfoto640) trims padding from the bottom up and stops at the
first non-white row, so it kept the whole page and fronts came out uncropped
(199 of 200 fronts in Paul's straps 016/017, 2026-11-07; backs were fine).

So find the paper's end from the TOP instead: the page starts with the paper,
and the padding begins at the first long run of near-white rows. Whatever
follows that run is driver junk and is dropped too.
"""

import numpy as np

NEAR_WHITE = 235       # min channel value counted as white (bill paper is ~205)
ROW_WHITE_FRAC = 0.95  # row is padding if this much of its middle is near-white
MIN_WHITE_RUN = 12     # consecutive white rows that mark the start of padding
MIN_CONTENT_ROWS = 40  # rows of content required before padding can start


def _white_rows(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    band = max(4, int(w * 0.05))          # ignore the far left/right columns
    core = img[:, band:w - band] if w > 4 * band else img
    if core.ndim == 3:
        core = core.min(axis=2)
    return (core >= NEAR_WHITE).mean(axis=1) >= ROW_WHITE_FRAC


def trim_padding(img: np.ndarray) -> np.ndarray:
    """Return the page cut at the start of its trailing white padding (and
    anything after it). Unchanged if there's no padding run."""
    white = _white_rows(img)
    content = np.flatnonzero(~white)
    if content.size == 0:
        return img
    # Search only after the content starts (a white-backed scanner may have
    # white rows above the bill) and after a minimum amount of it.
    start = int(content[0]) + MIN_CONTENT_ROWS
    if len(white) <= start + MIN_WHITE_RUN:
        return img
    run = np.convolve(white[start:].astype(np.int32),
                      np.ones(MIN_WHITE_RUN, np.int32), mode="valid")
    hits = np.flatnonzero(run == MIN_WHITE_RUN)
    if hits.size == 0:
        return img
    return img[: start + int(hits[0])]
