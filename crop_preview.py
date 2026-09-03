"""
Crop-settings preview context.

Supplies the eBay Crop Manager dialog with a live sample bill and the *real*
crop rectangles (computed by ProductionProcessor) so the preview box drawn on
the bill matches exactly what the pipeline would crop. Kept separate from the
dialog so any caller (crop_tool.py now, the main app later) can build one from
whatever bill it has on hand.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np


class CropPreviewContext:
    """Holds one aligned sample bill per side + its detections, and renders the
    crop rectangle for a given (side, region) under live config overrides.
    """

    def __init__(self, processor, front_img: Optional[np.ndarray] = None,
                 back_img: Optional[np.ndarray] = None):
        self.p = processor
        self.imgs = {'front': front_img, 'back': back_img}
        self.dets = {}
        for side, im in self.imgs.items():
            self.dets[side] = processor._detect_all_objects(im) if im is not None else {}

    def has_side(self, side: str) -> bool:
        return self.imgs.get(side) is not None

    def render(self, side: str, region: str, config: dict):
        """Return (bill_bgr, rect, crop_bgr).

        `config` is the dialog's current full config dict; its 'yolo_crops' and
        'crops' keys are applied temporarily so the rect reflects the live values.
        Any of the three return values may be None if unavailable.
        """
        img = self.imgs.get(side)
        if img is None:
            return None, None, None

        data = self.p.cfg.data
        saved = {k: data.get(k) for k in ('yolo_crops', 'crops')}
        try:
            for k in ('yolo_crops', 'crops'):
                if k in config and config[k] is not None:
                    data[k] = config[k]
            rect = self.p.crop_region_rect(img, self.dets[side], side, region)
        finally:
            for k, v in saved.items():
                if v is None:
                    data.pop(k, None)
                else:
                    data[k] = v

        crop = None
        if rect is not None:
            x1, y1, x2, y2 = rect
            if x2 > x1 and y2 > y1:
                crop = img[y1:y2, x1:x2]
        return img, rect, crop


def build_context_from_paths(processor, front_path, back_path=None) -> Optional[CropPreviewContext]:
    """Build a preview context from explicit front/back scan paths.

    Aligns them exactly as generate_crops() does (front checks flip; back follows
    the front's flip). Returns None if the front cannot be aligned.
    """
    import cv2

    front_img = back_img = None
    try:
        front_img, finfo = processor.yolo_aligner.align_image(str(front_path))
    except Exception:
        front_img = None
    if front_img is None:
        return None
    front_flipped = bool(finfo.get('flipped', False)) if isinstance(finfo, dict) else False
    if back_path:
        try:
            back_img, _ = processor.yolo_aligner.align_image(str(back_path), check_flip=False)
            if front_flipped and back_img is not None:
                back_img = cv2.rotate(back_img, cv2.ROTATE_180)
        except Exception:
            back_img = None
    return CropPreviewContext(processor, front_img, back_img)


def build_context_from_folder(processor, input_dir: Path) -> Optional[CropPreviewContext]:
    """Build a preview context from the first bill pair in a folder.

    Aligns the first detected front (and back, if present) exactly as the
    pipeline does. Returns None if no usable bill is found.
    """
    from process_production import ScannerFormatDetector
    import cv2

    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        return None
    try:
        _fmt, pairs = ScannerFormatDetector.find_pairs(input_dir)
    except Exception:
        pairs = []
    if not pairs:
        return None
    # Only verify the FIRST pair — verifying the whole folder would run YOLO on
    # every bill (30s+ on a 200-image folder) just to preview one sample.
    try:
        pairs = processor.verify_and_swap_pairs(pairs[:1])
    except Exception:
        pairs = pairs[:1]

    pair = pairs[0]
    return build_context_from_paths(processor, pair.front_path,
                                    getattr(pair, 'back_path', None))
