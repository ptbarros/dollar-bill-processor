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

# Color map for pattern highlights (CSS name -> BGR)
PATTERN_COLORS = {
    'purple': (128, 0, 128),    # Flipper digits
    'blue': (255, 0, 0),        # Binary
    'cyan': (255, 255, 0),      # Trinary
    'orange': (0, 165, 255),    # Radar pairs
    'coral': (80, 127, 255),    # Radar pair 2
    'gold': (0, 215, 255),      # Radar pair 3 / Quads
    'salmon': (114, 128, 250),  # Radar pair 4
    'magenta': (255, 0, 255),   # Repeater
    'yellow': (0, 255, 255),    # Solid/near-solid
    # 'lime'/'green' were retired (too close to the green serial digits) and now
    # render as cyan; kept as aliases so existing patterns/scripts don't break.
    'lime': (255, 255, 0),      # -> cyan (was Ladder green)
    'green': (255, 255, 0),     # -> cyan (was AI-usage green)
    'teal': (128, 128, 0),      # Pairs
    'red': (0, 0, 255),         # Broken/invalid
    'gray': (128, 128, 128),    # Muted/prefix
}

GAS_PUMP_FILTER = "__gas_pump__"
NONE_FILTER = "__none__"

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
    bbox_color=(0, 165, 255),
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
            padded-crop coordinates. Used both to offset digit coords and to draw
            the gas-pump bounding rectangle. If None, digit coords are assumed to
            already be padded-crop relative.
        bbox_color: BGR color for the gas-pump serial bounding rectangle.

    Returns:
        The same ``crop`` array, annotated in place.
    """
    is_gas_pump_mode = (overlay_filter == GAS_PUMP_FILTER)
    is_pattern_mode = (overlay_filter not in (GAS_PUMP_FILTER, NONE_FILTER))

    # Offset that maps tight-crop digit coords into the padded crop.
    if tight_box_rel is not None:
        off_x, off_y = tight_box_rel[0], tight_box_rel[1]
    else:
        off_x, off_y = 0, 0

    # Draw serial bounding box only in gas pump mode
    if is_gas_pump_mode and tight_box_rel is not None:
        bx1, by1, bx2, by2 = tight_box_rel
        cv2.rectangle(
            crop,
            (int(bx1 * zoom), int(by1 * zoom)),
            (int(bx2 * zoom), int(by2 * zoom)),
            bbox_color, 2,
        )

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

    # Store digit box info for drawing connectors and group boxes
    digit_centers = {}  # digit_idx -> (center_x, center_y)
    digit_rects = {}    # digit_idx -> (x1, y1, x2, y2)

    # Draw colored boxes for each digit.
    # Sort digit boxes left-to-right so position indices match pattern positions.
    digit_boxes = sorted(digit_boxes, key=lambda db: db['x1'])

    for idx, digit_box in enumerate(digit_boxes):
        # Convert digit coordinates to crop-relative, then apply zoom
        dx1 = int((digit_box['x1'] + off_x) * zoom)
        dy1 = int((digit_box['y1'] + off_y) * zoom)
        dx2 = int((digit_box['x2'] + off_x) * zoom)
        dy2 = int((digit_box['y2'] + off_y) * zoom)

        # Map digit_box index to pattern highlight index (skip letters)
        digit_idx = sum(1 for db in digit_boxes[:idx] if not db['is_letter'])

        # Store center and rect for connectors and group boxes
        if not digit_box['is_letter']:
            digit_centers[digit_idx] = ((dx1 + dx2) // 2, (dy1 + dy2) // 2)
            digit_rects[digit_idx] = (dx1, dy1, dx2, dy2)

        if is_gas_pump_mode:
            # Gas pump mode: show all boxes with deviation coloring
            if digit_box['is_letter']:
                color = (128, 128, 128)  # Gray
            elif digit_box['deviation'] >= gas_pump_threshold:
                color = (0, 0, 255)  # Red (BGR) - shifted
            else:
                color = (0, 255, 0)  # Green (BGR) - normal
            cv2.rectangle(crop, (dx1, dy1), (dx2, dy2), color, 2)

        elif is_pattern_mode:
            # Pattern mode: only show boxes for digits that match the pattern
            if not digit_box['is_letter'] and digit_idx < len(pattern_highlights):
                ph = pattern_highlights[digit_idx]
                if ph['highlights']:
                    first_highlight = ph['highlights'][0]
                    pattern_color = first_highlight.get('color', 'cyan')
                    color = PATTERN_COLORS.get(pattern_color, (0, 255, 0))
                    cv2.rectangle(crop, (dx1, dy1), (dx2, dy2), color, 2)

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
                conn_color = PATTERN_COLORS.get(conn.get('color', 'orange'), (0, 165, 255))
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
                    # Default arc connector: curved line above digits
                    pts = np.array([pt1, (mid_x, mid_y), pt2], np.int32)
                    cv2.polylines(crop, [pts], False, conn_color, 2, cv2.LINE_AA)

    # Draw group boxes (boxes spanning multiple digits)
    if is_pattern_mode and pattern_group_boxes:
        for gb in pattern_group_boxes:
            # Cast to int: on some Lua builds (e.g. the Windows package) these come
            # back as floats, and cv2 rejects a float thickness/coordinate -- which
            # would throw and abort the whole overlay, so year-note boxes silently
            # vanished.
            from_pos = int(gb.get('from', 0))
            to_pos = int(gb.get('to', 0))
            gb_color = PATTERN_COLORS.get(gb.get('color', 'magenta'), (255, 0, 255))
            thickness = int(gb.get('thickness', 3))

            if from_pos in digit_rects and to_pos in digit_rects:
                r1 = digit_rects[from_pos]
                r2 = digit_rects[to_pos]

                # Combine rects: min x1/y1, max x2/y2 with padding
                padding = 4
                gx1 = int(min(r1[0], r2[0]) - padding)
                gy1 = int(min(r1[1], r2[1]) - padding)
                gx2 = int(max(r1[2], r2[2]) + padding)
                gy2 = int(max(r1[3], r2[3]) + padding)

                cv2.rectangle(crop, (gx1, gy1), (gx2, gy2), gb_color, thickness)

    return crop
