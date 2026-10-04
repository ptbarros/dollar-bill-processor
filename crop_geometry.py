"""
Drag <-> crop-config inversion for the interactive Crop Manager canvas.

The crop tool draws each seal / serial crop as a draggable box on a sample bill.
When the user drags or resizes a box we have to turn the new rectangle back into
the SAME ``yolo_crops`` knobs the numeric spinboxes drive, so a drag and a spinbox
edit are two views of one setting and the real pipeline crops identically.

Two crop shapes exist, both anchored to a YOLO detection (so they stay
shift-safe across scans):

* **offset+min** (front seal, left serial, right serial): the rect is the padded
  anchor union, shifted by ``offset_x`` / ``offset_y`` (positive x = right,
  positive y = UP), then expanded *symmetrically* to at least
  ``min_width`` x ``min_height``. So the final centre is the base centre shifted
  by the offset, and the size is ``max(natural, min)``. Inversion:
      offset_x  = dragged_cx - base_cx
      offset_y  = base_cy   - dragged_cy      (positive = up)
      min_width = dragged_w  ; min_height = dragged_h
  Because these are *minimums*, a box can't be dragged smaller than its content
  (the min just stops binding) -- that's the model, not a bug.

* **width/height** (back seal): anchored at the back plate's left edge / bottom;
  ``x1 = bp_x1 + offset_x``, ``y2 = bp_y2 - offset_y``, then width/height extend
  right and up. Inversion:
      offset_x = dragged_x1 - base_x1   ; offset_y = base_y2 - dragged_y2
      width    = dragged_w              ; height   = dragged_h

Everything is expressed against a **base rect** -- the rect the existing
preview renderer produces with that region's offsets (and, for offset+min, its
mins) zeroed. That keeps this module decoupled from the detection internals:
give it a ``render(config) -> rect`` callable and the dragged rect, get config back.
"""

from __future__ import annotations

import copy
from typing import Callable, Optional, Tuple

Rect = Tuple[int, int, int, int]

# region (as used by crop_region_rect / the dialog rows) -> yolo_crops sub-key.
# Only these are draggable; thirds (left/center/right) are dynamic/auto-extending
# and 'full' has no knobs, so they are display-only on the canvas.
_OFFSET_MIN_KEYS = {"front_seal", "serial_left", "serial_right"}
_WIDTH_HEIGHT_KEYS = {"back_seal"}


def region_config_key(side: str, region: str) -> Optional[str]:
    """Map a (side, region) crop to its ``yolo_crops`` sub-key, or None if the
    crop is not draggable (thirds / full / percentage fallbacks)."""
    if region == "seal":
        return "front_seal" if side == "front" else "back_seal"
    if region in ("serial_left", "serial_right"):
        return region
    return None


def is_draggable(side: str, region: str) -> bool:
    return region_config_key(side, region) is not None


def _zeroed_config(config: dict, key: str) -> dict:
    """A deep copy of ``config`` with the given region's offsets (and, for the
    offset+min shapes, its mins) set to zero, so a render yields the base rect."""
    cfg = copy.deepcopy(config or {})
    yc = cfg.setdefault("yolo_crops", {})
    sub = dict(yc.get(key, {}))
    sub["offset_x"] = 0
    sub["offset_y"] = 0
    if key in _OFFSET_MIN_KEYS:
        sub["min_width"] = 0
        sub["min_height"] = 0
    yc[key] = sub
    return cfg


def base_rect(render: Callable[[dict], Optional[Rect]], config: dict,
              key: str) -> Optional[Rect]:
    """Render the region's base rect (offsets/mins zeroed).

    ``render`` takes a full config dict and returns (x1,y1,x2,y2) or None
    (e.g. ``lambda c: ctx.render(side, region, c)[1]``).
    """
    return render(_zeroed_config(config, key))


def invert_drag(base: Rect, dragged: Rect, key: str) -> dict:
    """Return the updated ``yolo_crops[key]`` values for a dragged rectangle.

    ``base`` is the region's base rect (see :func:`base_rect`); ``dragged`` is the
    rectangle the user left the box at. Result keys depend on the crop shape.
    """
    bx1, by1, bx2, by2 = base
    dx1, dy1, dx2, dy2 = dragged
    dw = max(0, dx2 - dx1)
    dh = max(0, dy2 - dy1)

    if key in _WIDTH_HEIGHT_KEYS:
        return {
            "offset_x": int(round(dx1 - bx1)),
            "offset_y": int(round(by2 - dy2)),
            "width": int(round(dw)),
            "height": int(round(dh)),
        }

    # offset + symmetric min-size shape
    base_cx = (bx1 + bx2) / 2.0
    base_cy = (by1 + by2) / 2.0
    drag_cx = (dx1 + dx2) / 2.0
    drag_cy = (dy1 + dy2) / 2.0
    return {
        "offset_x": int(round(drag_cx - base_cx)),
        "offset_y": int(round(base_cy - drag_cy)),   # positive = up
        "min_width": int(round(dw)),
        "min_height": int(round(dh)),
    }


def invert_fixed(dragged: Rect, img_w: int, img_h: int) -> dict:
    """A hand-placed 'fixed' box -> {x,y,w,h} as fractions of the bill.

    Used for seal/serial crops on denominations the model can't anchor: the box is
    positioned freely and stored relative to the bill so it holds across scans.
    """
    dx1, dy1, dx2, dy2 = dragged
    W = float(img_w or 1)
    H = float(img_h or 1)
    return {
        "x": max(0.0, min(dx1 / W, 1.0)),
        "y": max(0.0, min(dy1 / H, 1.0)),
        "w": max(0.0, min((dx2 - dx1) / W, 1.0)),
        "h": max(0.0, min((dy2 - dy1) / H, 1.0)),
    }


def set_fixed(config: dict, key: str, dragged: Rect, img_w: int, img_h: int) -> dict:
    """Switch a region to fixed mode with the dragged box, in place. Returns config."""
    return apply_updates(config, key,
                         {"mode": "fixed",
                          "fixed": invert_fixed(dragged, img_w, img_h)})


def clear_fixed(config: dict, key: str) -> dict:
    """Return a region to anchored mode (drop mode/fixed), in place."""
    yc = config.setdefault("yolo_crops", {})
    sub = dict(yc.get(key, {}))
    sub.pop("mode", None)
    sub.pop("fixed", None)
    yc[key] = sub
    return config


def is_fixed(config: dict, key: str) -> bool:
    sub = (config.get("yolo_crops") or {}).get(key) or {}
    return sub.get("mode") == "fixed" and isinstance(sub.get("fixed"), dict)


def apply_updates(config: dict, key: str, updates: dict) -> dict:
    """Merge ``updates`` into ``config['yolo_crops'][key]`` in place, returning it."""
    yc = config.setdefault("yolo_crops", {})
    sub = dict(yc.get(key, {}))
    sub.update(updates)
    yc[key] = sub
    return config


# --- thirds (left / center / right) -------------------------------------
# Thirds crops are full-height; only their vertical boundary edges move, mapping
# to the per-side overlap knobs yolo_crops.thirds.<side>.{left_inner, right_inner,
# center_left, center_right} (positive = grow toward the bill centre). Signs match
# process_production._thirds_rects.
THIRDS_REGIONS = {"left", "center", "right"}


def thirds_edges(region: str) -> Set[str]:
    """Which box edges are draggable for a thirds crop: the left crop's right edge,
    the right crop's left edge, both of the centre crop's."""
    return {"r"} if region == "left" else {"l"} if region == "right" else {"l", "r"}


def _zeroed_thirds(config: dict, side: str) -> dict:
    cfg = copy.deepcopy(config or {})
    thirds = cfg.setdefault("yolo_crops", {}).setdefault("thirds", {})
    thirds[side] = {"left_inner": 0, "right_inner": 0,
                    "center_left": 0, "center_right": 0}
    return cfg


def base_thirds_rect(render: Callable[[dict], Optional[Rect]], config: dict,
                     side: str) -> Optional[Rect]:
    """The thirds rect with all overlap knobs zeroed (the natural boundary)."""
    return render(_zeroed_thirds(config, side))


def invert_thirds(base: Rect, dragged: Rect, region: str) -> dict:
    """Return the updated thirds overlap knobs for a dragged boundary edge."""
    bx1, _, bx2, _ = base
    dx1, _, dx2, _ = dragged
    if region == "left":
        return {"left_inner": int(round(dx2 - bx2))}
    if region == "right":
        return {"right_inner": int(round(bx1 - dx1))}
    return {"center_left": int(round(bx1 - dx1)),
            "center_right": int(round(dx2 - bx2))}


def apply_thirds(config: dict, side: str, updates: dict) -> dict:
    """Merge thirds overlap updates into yolo_crops.thirds.<side>, in place."""
    thirds = config.setdefault("yolo_crops", {}).setdefault("thirds", {})
    sub = dict(thirds.get(side, {}))
    sub.update(updates)
    thirds[side] = sub
    return config


def config_from_drag(render: Callable[[dict], Optional[Rect]], config: dict,
                     side: str, region: str, dragged: Rect) -> Optional[dict]:
    """One-shot: given the live config, the dragged rect and a render callable,
    return the ``yolo_crops[key]`` updates (or None if the region isn't draggable
    or its base rect can't be computed)."""
    key = region_config_key(side, region)
    if key is None:
        return None
    base = base_rect(render, config, key)
    if base is None:
        return None
    return invert_drag(base, dragged, key)
