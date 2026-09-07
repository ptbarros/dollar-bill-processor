"""Shared serial-region overlay drawing.

Single source of truth for drawing the pattern / gas-pump overlay onto a
cropped serial-number region. Used both by the GUI preview panel
(``gui/preview_panel._generate_serial_region_crops``) and by the crop pipeline
(``process_production.generate_crops``) so the saved overlay crop matches
exactly what the user sees in the preview.

The tricky, historically bug-prone bits live here now:
  * group-box coords/thickness are cast to int (some Lua builds return floats,
    which make cv2 throw and abort the whole overlay -- year-note boxes used to
    vanish on Windows for this reason).
"""

import cv2
import numpy as np

# Color map for overlay highlights (name -> BGR).
#
# Colors are chosen for CONTRAST/VISIBILITY on the bill, NOT to signal pattern
# type (the Patterns column already identifies the pattern). All of the "core"
# colors below were contrast-tested against the measured bill background -- cream
# paper RGB(231,231,199) and green serial ink RGB(75,143,68) -- for strong
# perceptual contrast (CIELAB deltaE) against BOTH, and for being mutually
# distinguishable when several appear at once (e.g. radar pairs).
#
# Recommended rotation order (most visible + most distinct first):
#   blue -> orange -> magenta -> red -> purple -> hotpink   (+ black for max
#   contrast on busy backgrounds; gray for muted prefix letters).
#
# Weak colors on this background (yellow/gold on cream, cyan/teal/green on the
# green ink, white on paper) are RETIRED: kept as keys so existing/user/AI
# patterns don't break, but aliased to the nearest strong color so they still
# render clearly instead of fading out.
# Default BGR for each user-editable palette slot. This is the SINGLE SOURCE of
# truth for overlay colors -- the pattern-preview widget derives its colors from
# here too (via bgr_hex/hex_to_bgr), so there's no second palette to keep in sync.
# The Overlay Colors tool lets the user override any slot; overrides are applied
# with set_overlay_overrides() and PATTERN_COLORS (incl. aliases) is rebuilt.
_BASE_COLORS = {
    'blue':    (220, 60, 30),    # royalblue  (RGB 30,60,220)  strongest
    'orange':  (0, 140, 255),    #            (RGB 255,140,0)
    'magenta': (200, 0, 255),    #            (RGB 255,0,200)
    'red':     (30, 30, 230),    #            (RGB 230,30,30)
    'purple':  (230, 60, 160),   #            (RGB 160,60,230)
    'hotpink': (150, 60, 255),   #            (RGB 255,60,150)
    'black':   (10, 10, 10),     # max contrast on any background
    'gray':    (128, 128, 128),  # muted / prefix letters (intentionally low-key)
    'charcoal': (60, 60, 60),    # darker muted -- good for an "excluded" X mark
}

# Editable slots shown in the Overlay Colors tool, in display order.
EDITABLE_SLOTS = ['blue', 'orange', 'magenta', 'red', 'purple', 'hotpink',
                  'black', 'gray', 'charcoal']

# Retired/weak color names kept as keys (so existing/user/AI patterns don't
# break) but aliased to the nearest strong slot -- follows that slot's override.
_ALIASES = {
    'pink': 'hotpink',
    'cyan': 'blue', 'teal': 'blue', 'lime': 'blue', 'green': 'blue',
    'yellow': 'orange', 'gold': 'orange', 'amber': 'orange',
    'coral': 'red', 'salmon': 'hotpink', 'white': 'black',
}

_overrides = {}  # slot name -> BGR tuple (user overrides)
PATTERN_COLORS = {}  # working palette (base + overrides + aliases); rebuilt below


def hex_to_bgr(h):
    """'#rrggbb' -> (b, g, r)."""
    h = h.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def bgr_hex(bgr):
    """(b, g, r) -> '#rrggbb'."""
    b, g, r = bgr
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def default_slot_hex(name):
    """Factory-default hex for a slot (for the tool's Reset)."""
    return bgr_hex(_BASE_COLORS[name])


def build_palette(overrides_hex):
    """Return a full palette dict {name: BGR} (base + overrides + aliases) WITHOUT
    mutating the global PATTERN_COLORS. Used by the Overlay Colors tool to preview
    hypothetical colors before the user commits them."""
    base = dict(_BASE_COLORS)
    for name, h in (overrides_hex or {}).items():
        if name in _BASE_COLORS and h:
            base[name] = hex_to_bgr(h)
    full = dict(base)
    for alias, target in _ALIASES.items():
        full[alias] = base[target]          # aliases follow their target's override
    return full


def _rebuild_palette():
    """Rebuild the global PATTERN_COLORS from base + committed user overrides."""
    global PATTERN_COLORS
    PATTERN_COLORS = build_palette({n: bgr_hex(bgr) for n, bgr in _overrides.items()})


def set_overlay_overrides(overrides_hex):
    """Apply user overlay-color overrides ({slot: '#rrggbb'}) and rebuild the
    palette. Unknown slots / empty values are ignored. Call at startup and again
    whenever the Overlay Colors tool applies changes."""
    global _overrides
    _overrides = {n: hex_to_bgr(h) for n, h in (overrides_hex or {}).items()
                  if n in _BASE_COLORS and h}
    _rebuild_palette()


_rebuild_palette()

GAS_PUMP_FILTER = "__gas_pump__"
NONE_FILTER = "__none__"


# Muted/neutral colors are exempt from the rotation -- they carry "de-emphasized
# / excluded" meaning (e.g. a gray or charcoal X mark) and must stay that color.
_ROTATION_EXEMPT = ('gray', 'charcoal')

# Contrast-optimized rotation order (most visible + most distinct first). Color
# is assigned by first-appearance within an overlay, NOT by pattern type. Muted
# colors (_ROTATION_EXEMPT) are reserved for muted/prefix/excluded and never
# rotated.
#
# Order is tuned so (a) the first four are the maximally-distinct set for the
# common <=4-group patterns, and (b) adjacent slots stay far apart in CIELAB
# (min adjacent deltaE ~92) -- the two closest pairs, magenta+hotpink and
# blue+purple, are never neighbors, and black sits between purple and hotpink as
# a separator. So similar colors don't land next to each other on the bill.
_ROTATION = ('blue', 'orange', 'magenta', 'red', 'purple', 'black', 'hotpink')


def _build_color_rotation(highlights, connectors, group_boxes):
    """Map each distinct requested color name (in first-seen order) to a strong
    rotation color, so drawn colors are always high-contrast and distinct. 'gray'
    is passed through untouched (it stays muted)."""
    seen = []

    def note(nm):
        if nm and nm not in _ROTATION_EXEMPT and nm not in seen:
            seen.append(nm)

    for ph in (highlights or []):
        for h in ph.get('highlights', []):
            note(h.get('color'))
    for c in (connectors or []):
        note(c.get('color'))
    for gb in (group_boxes or []):
        note(gb.get('color'))

    return {nm: PATTERN_COLORS[_ROTATION[i % len(_ROTATION)]]
            for i, nm in enumerate(seen)}


def _quad_bezier_points(p0, ctrl, p1, segments=24):
    """Sample a quadratic Bezier (p0 -> ctrl -> p1) into an int32 point array for
    cv2.polylines, giving a smooth arc instead of an angular 3-point tent."""
    pts = []
    for i in range(segments + 1):
        t = i / segments
        mt = 1.0 - t
        x = mt * mt * p0[0] + 2 * mt * t * ctrl[0] + t * t * p1[0]
        y = mt * mt * p0[1] + 2 * mt * t * ctrl[1] + t * t * p1[1]
        pts.append((int(round(x)), int(round(y))))
    return np.array(pts, np.int32)

# eBay rejects uploads whose shorter side is under 500px. The serial overlay
# crop is naturally wide-and-short (~786x154), so its height falls under this.
EBAY_MIN_DIMENSION = 500


def pad_to_min(img, min_size=EBAY_MIN_DIMENSION, color=(0, 0, 0)):
    """Center ``img`` on a solid canvas so both sides are at least ``min_size``.

    Used to lift the wide-and-short serial overlay crop over eBay's 500px minimum
    dimension. Black padding also keeps the visual emphasis on the serial. Returns
    the original array unchanged when it already meets the minimum.
    """
    h, w = img.shape[:2]
    out_h = max(h, min_size)
    out_w = max(w, min_size)
    if out_h == h and out_w == w:
        return img

    if img.ndim == 3:
        canvas = np.zeros((out_h, out_w, img.shape[2]), dtype=img.dtype)
        canvas[:] = color[:img.shape[2]]
    else:
        canvas = np.zeros((out_h, out_w), dtype=img.dtype)
        canvas[:] = color[0]

    y0 = (out_h - h) // 2
    x0 = (out_w - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    return canvas


def resolve_overlay_filter(overlay_filter, matched_patterns, pattern_override=None):
    """Pick which overlay to render for a bill in the crop pipeline.

    Order of preference:
      1. An explicit ``pattern_override`` (right-click "Set Pattern..."), if the
         bill actually matched it.
      2. The caller-supplied ``overlay_filter`` if it is a real matched pattern.
      3. The top (first) matched fancy pattern, ignoring GAS_PUMP.
      4. Gas-pump overlay if the bill only matched GAS_PUMP.
      5. ``NONE_FILTER`` (plain serial crop) as a last resort.
    """
    matched = [p for p in (matched_patterns or []) if p]
    non_gp = [p for p in matched if p != 'GAS_PUMP']

    if pattern_override and pattern_override in matched:
        return pattern_override
    if overlay_filter in matched and overlay_filter not in (GAS_PUMP_FILTER, NONE_FILTER):
        return overlay_filter
    if non_gp:
        return non_gp[0]
    if 'GAS_PUMP' in matched:
        return GAS_PUMP_FILTER
    return NONE_FILTER


def draw_serial_overlay(
    crop,
    digit_boxes,
    *,
    zoom,
    overlay_filter,
    serial,
    matched_patterns,
    pattern_engine,
    gas_pump_threshold,
    tight_box_rel=None,
):
    """Draw the pattern / gas-pump overlay onto ``crop`` in place and return it.

    Args:
        crop: BGR image already cropped to the padded serial region and already
            scaled by ``zoom``.
        digit_boxes: list of dicts from ``analyze_gas_pump_digits`` (coords are
            relative to the TIGHT serial crop, in unzoomed pixels). Each dict has
            x1/y1/x2/y2/is_letter/deviation.
        zoom: scale factor already applied to ``crop``.
        overlay_filter: ``GAS_PUMP_FILTER`` | ``NONE_FILTER`` | pattern internal
            name.
        serial: serial string used to compute pattern highlights.
        matched_patterns: list of pattern names the bill matched.
        pattern_engine: engine exposing ``get_digit_highlights``.
        gas_pump_threshold: pixel deviation threshold for gas-pump coloring.
        tight_box_rel: (x1, y1, x2, y2) of the tight serial box in unzoomed
            padded-crop coordinates, used to offset digit coords. If None, digit
            coords are assumed to already be padded-crop relative.

    Returns:
        The crop array, annotated. It may be a TALLER array than the one passed
        in (grown upward to give nested connector arcs headroom), so callers must
        use the return value rather than the array they passed.
    """
    is_gas_pump_mode = (overlay_filter == GAS_PUMP_FILTER)
    is_pattern_mode = (overlay_filter not in (GAS_PUMP_FILTER, NONE_FILTER))

    # Offset that maps tight-crop digit coords into the padded crop.
    if tight_box_rel is not None:
        off_x, off_y = tight_box_rel[0], tight_box_rel[1]
    else:
        off_x, off_y = 0, 0


    # Get pattern-based digit highlights for specific pattern mode
    pattern_highlights = []
    pattern_connectors = []
    pattern_group_boxes = []
    if is_pattern_mode and serial:
        patterns_for_highlights = [overlay_filter] if overlay_filter in (matched_patterns or []) else []
        if patterns_for_highlights:
            viz_data = pattern_engine.get_digit_highlights(serial, patterns_for_highlights)
            pattern_highlights = viz_data.get('highlights', [])
            pattern_connectors = viz_data.get('connectors', [])
            pattern_group_boxes = viz_data.get('group_boxes', [])

    # Remap the colors a pattern REQUESTED onto the contrast-optimized rotation, so
    # the drawn colors are always strong and mutually distinct regardless of which
    # names the pattern author picked (color = visibility, not a pattern-type
    # signal). Distinct sub-groups within one pattern stay distinct.
    color_map = _build_color_rotation(
        pattern_highlights, pattern_connectors, pattern_group_boxes)

    def _resolve(name):
        if name in color_map:
            return color_map[name]
        return PATTERN_COLORS.get(name, PATTERN_COLORS['blue'])

    # Store digit box info for drawing connectors and group boxes
    digit_centers = {}  # digit_idx -> (center_x, center_y)
    digit_rects = {}    # digit_idx -> (x1, y1, x2, y2)

    # Draw colored boxes for each digit.
    # Sort digit boxes left-to-right so position indices match pattern positions.
    digit_boxes = sorted(digit_boxes, key=lambda db: db['x1'])

    # --- Reserve top headroom for nested connector arcs --------------------
    # Connector arcs are lifted by nesting depth so concentric pairs (radars)
    # stack as clean nested curves with short stubs to each digit, matching the
    # DigitPreviewWidget preview. Lifted arcs need room ABOVE the digits, so grow
    # the crop upward (replicating the bill background, not a black bar) and shift
    # every drawn y down by top_pad. The taller crop is RETURNED -- callers must
    # use the return value, not the passed-in array.
    ARC_H, ARC_STEP, ARC_GAP, ARC_MARGIN = 14, 12, 4, 6

    conn_spans = []
    if is_pattern_mode and pattern_connectors:
        for _conn in pattern_connectors:
            if 'positions' in _conn:
                _a, _b = _conn['positions']
            else:
                _a, _b = _conn.get('from', 0), _conn.get('to', 0)
            conn_spans.append((min(_a, _b), max(_a, _b)))

    def _nest_level(lo, hi):
        """How many other arcs this one contains (its concentric depth)."""
        return sum(1 for (l2, h2) in conn_spans
                   if lo <= l2 and h2 <= hi and (l2, h2) != (lo, hi))

    top_pad = 0
    if conn_spans:
        tops = {}
        _di = 0
        for db in digit_boxes:
            if not db['is_letter']:
                # -8 conservatively accounts for the display box padding (bpad).
                tops[_di] = int((db['y1'] + off_y) * zoom) - 8
                _di += 1
        min_peak = None
        for (lo, hi) in conn_spans:
            if lo in tops and hi in tops:
                peak = min(tops[lo], tops[hi]) - ARC_GAP - _nest_level(lo, hi) * ARC_STEP - ARC_H
                min_peak = peak if min_peak is None else min(min_peak, peak)
        if min_peak is not None and min_peak < ARC_MARGIN:
            top_pad = ARC_MARGIN - min_peak
    if top_pad > 0:
        crop = cv2.copyMakeBorder(crop, top_pad, 0, 0, 0, cv2.BORDER_REPLICATE)

    for idx, digit_box in enumerate(digit_boxes):
        # Convert digit coordinates to crop-relative, then apply zoom
        dx1 = int((digit_box['x1'] + off_x) * zoom)
        dy1 = int((digit_box['y1'] + off_y) * zoom) + top_pad
        dx2 = int((digit_box['x2'] + off_x) * zoom)
        dy2 = int((digit_box['y2'] + off_y) * zoom) + top_pad

        # The digit boxes come from gas-pump segmentation, which are TIGHT glyph
        # contours (needed for accurate baseline/deviation). For DISPLAY only, give
        # the drawn box a little breathing room so it doesn't hug the digit. This is
        # decoupled from the gas-pump measurement, which still uses the tight box.
        ch, cw = crop.shape[:2]
        bpad = max(2, min(8, int(round((dy2 - dy1) * 0.14))))
        pdx1, pdy1 = max(0, dx1 - bpad), max(0, dy1 - bpad)
        pdx2, pdy2 = min(cw - 1, dx2 + bpad), min(ch - 1, dy2 + bpad)

        # Map digit_box index to pattern highlight index (skip letters)
        digit_idx = sum(1 for db in digit_boxes[:idx] if not db['is_letter'])

        # Store center (true center) and rect (padded, so group boxes + arcs line
        # up with the drawn boxes) for connectors and group boxes.
        if not digit_box['is_letter']:
            digit_centers[digit_idx] = ((dx1 + dx2) // 2, (dy1 + dy2) // 2)
            digit_rects[digit_idx] = (pdx1, pdy1, pdx2, pdy2)

        if is_gas_pump_mode:
            # Gas pump mode: box ONLY the misaligned digit(s), in red. Aligned
            # digits and the prefix/suffix letters are left un-boxed so the shifted
            # digit stands out on its own -- its box sits visibly higher/lower than
            # its neighbors. (Boxing every digit green + the letters gray was just
            # clutter for what is really a one- or two-digit signal.) Lower the Gas
            # Pump slider to reveal borderline digits.
            if (not digit_box['is_letter']
                    and digit_box['deviation'] >= gas_pump_threshold):
                cv2.rectangle(crop, (pdx1, pdy1), (pdx2, pdy2),
                              PATTERN_COLORS['red'], 2)

        elif is_pattern_mode:
            # Pattern mode: only show boxes for digits that match the pattern
            if not digit_box['is_letter'] and digit_idx < len(pattern_highlights):
                ph = pattern_highlights[digit_idx]
                if ph['highlights']:
                    first_highlight = ph['highlights'][0]
                    color = _resolve(first_highlight.get('color', 'blue'))
                    style = (first_highlight.get('style') or 'box').lower()
                    # 'x' = X only (excluded digit), 'boxed_x' = box + X, else box.
                    if style in ('x', 'boxed_x'):
                        if style == 'boxed_x':
                            cv2.rectangle(crop, (pdx1, pdy1), (pdx2, pdy2), color, 2)
                        cv2.line(crop, (pdx1 + 2, pdy1 + 2), (pdx2 - 2, pdy2 - 2), color, 2, cv2.LINE_AA)
                        cv2.line(crop, (pdx1 + 2, pdy2 - 2), (pdx2 - 2, pdy1 + 2), color, 2, cv2.LINE_AA)
                    else:
                        cv2.rectangle(crop, (pdx1, pdy1), (pdx2, pdy2), color, 2)

    # Draw connector lines for relational patterns (e.g., RADAR pairs)
    if is_pattern_mode and pattern_connectors:
        for conn in pattern_connectors:
            # Support both formats: {positions: [a, b]} and {from: a, to: b}
            if 'positions' in conn:
                pos1, pos2 = conn['positions']
            else:
                pos1 = conn.get('from', 0)
                pos2 = conn.get('to', 0)

            if pos1 in digit_centers and pos2 in digit_centers:
                pt1 = digit_centers[pos1]
                pt2 = digit_centers[pos2]
                conn_color = _resolve(conn.get('color', 'orange'))
                conn_style = conn.get('style', 'arc')

                # Calculate arc - height proportional to distance, but capped
                mid_x = (pt1[0] + pt2[0]) // 2
                distance = abs(pt2[0] - pt1[0])
                arc_height = min(35, max(15, distance // 8))
                mid_y = min(pt1[1], pt2[1]) - arc_height

                if conn_style in ('broken', 'dashed'):
                    # Broken/dashed pair: X marks near each digit (no line)
                    x_size = 5
                    x1_pos = pt1[0] + 15
                    x1_y = pt1[1] - 10
                    cv2.line(crop, (x1_pos - x_size, x1_y - x_size), (x1_pos + x_size, x1_y + x_size), conn_color, 2, cv2.LINE_AA)
                    cv2.line(crop, (x1_pos - x_size, x1_y + x_size), (x1_pos + x_size, x1_y - x_size), conn_color, 2, cv2.LINE_AA)
                    x2_pos = pt2[0] - 15
                    x2_y = pt2[1] - 10
                    cv2.line(crop, (x2_pos - x_size, x2_y - x_size), (x2_pos + x_size, x2_y + x_size), conn_color, 2, cv2.LINE_AA)
                    cv2.line(crop, (x2_pos - x_size, x2_y + x_size), (x2_pos + x_size, x2_y - x_size), conn_color, 2, cv2.LINE_AA)
                elif conn_style == 'line':
                    cv2.line(crop, pt1, pt2, conn_color, 2, cv2.LINE_AA)
                elif conn_style == 'bracket':
                    bracket_y = max(pt1[1], pt2[1]) + 10
                    cv2.line(crop, (pt1[0], pt1[1] + 5), (pt1[0], bracket_y), conn_color, 2, cv2.LINE_AA)
                    cv2.line(crop, (pt1[0], bracket_y), (pt2[0], bracket_y), conn_color, 2, cv2.LINE_AA)
                    cv2.line(crop, (pt2[0], bracket_y), (pt2[0], pt2[1] + 5), conn_color, 2, cv2.LINE_AA)
                elif conn_style == 'arrow':
                    cv2.arrowedLine(crop, pt1, pt2, conn_color, 2, cv2.LINE_AA, tipLength=0.15)
                else:
                    # Default arc connector: a smooth curve arching ABOVE the
                    # digits, matching the DigitPreviewWidget preview. Endpoints
                    # anchor to the digit tops (not glyph centers) so the arc never
                    # slices through them. Lift by nesting depth so concentric arcs
                    # stack as clean nested curves instead of converging, with short
                    # stubs tying each lifted endpoint back down to its box. The
                    # crop was grown upward above to give these arcs room.
                    r1 = digit_rects.get(pos1)
                    r2 = digit_rects.get(pos2)
                    if r1 and r2:
                        x1c = (r1[0] + r1[2]) // 2
                        x2c = (r2[0] + r2[2]) // 2
                        top_y = min(r1[1], r2[1])
                    else:  # fallback: no rects -> use centers
                        x1c, x2c = pt1[0], pt2[0]
                        top_y = min(pt1[1], pt2[1])
                    level = _nest_level(min(pos1, pos2), max(pos1, pos2))
                    end_y = max(1, top_y - ARC_GAP - level * ARC_STEP)
                    peak_y = max(1, end_y - ARC_H)
                    box_top = top_y - 1
                    if end_y < box_top:  # stubs from each box top up to the arc
                        cv2.line(crop, (x1c, box_top), (x1c, end_y), conn_color, 2, cv2.LINE_AA)
                        cv2.line(crop, (x2c, box_top), (x2c, end_y), conn_color, 2, cv2.LINE_AA)
                    arc_pts = _quad_bezier_points(
                        (x1c, end_y), (mid_x, peak_y), (x2c, end_y))
                    cv2.polylines(crop, [arc_pts], False, conn_color, 2, cv2.LINE_AA)

    # Draw group boxes (boxes spanning multiple digits)
    if is_pattern_mode and pattern_group_boxes:
        for gb in pattern_group_boxes:
            # Cast to int: on some Lua builds (e.g. the Windows package) these come
            # back as floats, and cv2 rejects a float thickness/coordinate -- which
            # would throw and abort the whole overlay, so year-note boxes silently
            # vanished.
            from_pos = int(gb.get('from', 0))
            to_pos = int(gb.get('to', 0))
            gb_color = _resolve(gb.get('color', 'magenta'))
            thickness = int(gb.get('thickness', 3))

            if from_pos in digit_rects and to_pos in digit_rects:
                r1 = digit_rects[from_pos]
                r2 = digit_rects[to_pos]

                # digit_rects are already display-padded; add a small extra gap so
                # the group box sits just outside the per-digit boxes. Clamp to the
                # crop so the top/left edge never falls off-image.
                ch, cw = crop.shape[:2]
                padding = 2
                gx1 = max(0, int(min(r1[0], r2[0]) - padding))
                gy1 = max(0, int(min(r1[1], r2[1]) - padding))
                gx2 = min(cw - 1, int(max(r1[2], r2[2]) + padding))
                gy2 = min(ch - 1, int(max(r1[3], r2[3]) + padding))

                cv2.rectangle(crop, (gx1, gy1), (gx2, gy2), gb_color, thickness)

    return crop
