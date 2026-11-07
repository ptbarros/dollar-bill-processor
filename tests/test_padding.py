"""WinRT pages: white padding after paper end, with dark sensor-edge columns
running through it on the front side (Paul's strap 017 front crops)."""
import numpy as np

from dollar_detective.scanner import billcrop
from dollar_detective.scanner.padding import trim_padding

BG, BILL = (172, 167, 160), (200, 209, 205)


def raw_page(dark_edges, jpeg_speckle=False):
    W, H = 2100, 1415                       # 7.0 x 4.72 in at 300 dpi
    img = np.full((H, W, 3), 255, np.uint8)
    img[:830] = BG                           # grey backing while paper passes
    img[20:803, 140:140 + 1842] = BILL       # 6.14 x 2.61 in bill
    if dark_edges:
        img[:, :12] = 30
        img[:, -12:] = 30
    if jpeg_speckle:
        rng = np.random.default_rng(0)
        ys, xs = rng.integers(830, H, 400), rng.integers(0, W, 400)
        img[ys, xs] = 240
    return img


def test_trims_padding_despite_dark_edge_columns():
    img = raw_page(dark_edges=True, jpeg_speckle=True)
    assert billcrop._trim_white_padding(img).shape[0] == 1415   # the old bug
    assert trim_padding(img).shape[0] == 830


def test_front_crop_is_bill_sized_after_trim():
    img = raw_page(dark_edges=True)
    out, found = billcrop.crop_item(trim_padding(img), margin_px=6)
    h, w = out.shape[:2]
    assert found
    assert abs(w - (1842 + 12)) < 15 and abs(h - (783 + 12)) < 15


def test_no_padding_is_untouched():
    img = raw_page(dark_edges=False)[:830]
    assert trim_padding(img).shape[0] == 830
