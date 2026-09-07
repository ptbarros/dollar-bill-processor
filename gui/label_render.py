"""Label rendering — the single source of truth for how a printed label looks.

Phase 1 of the label-printing tool. The on-screen PREVIEW and the exported files
are produced by the *same* drawing code (`draw_label`), so what the user sees is
exactly what prints.

Physical target: a Rollo 2.0 x 1.0 inch label, zero margins, 10pt text.

Content model (two layers):
  * A TEMPLATE (global, per label size) is an ordered list of FIELDS. Each field
    pulls a bill value (serial / series / pattern / note / catalog / position),
    has a CAPTION the user sets once (e.g. "SERIES " -> "Ser: "), a same-line
    flag (share the previous row or start a new one), and an enabled flag.
    Editing the template changes every label at once. Empty values are skipped
    so a bill with no note doesn't print a blank line.
  * A per-bill OVERRIDE (`LabelData.lines`) is the one-off escape hatch: freeform
    text that prints verbatim for that bill only. Reset clears it, falling back
    to the template.

Long lines WRAP to the next row; the only overflow warning is vertical -- too
many rows to fit the label height.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from PySide6.QtCore import Qt, QSizeF, QMarginsF
from PySide6.QtGui import (
    QFont, QFontMetricsF, QColor, QPainter, QImage,
    QPdfWriter, QPageSize, QPageLayout,
)

# --- Physical label geometry (inches) ------------------------------------
LABEL_W_IN = 2.0
LABEL_H_IN = 1.0
MARGIN_IN = 0.05      # matches the docx 72-twip L/R indent
TOP_PAD_IN = 0.055    # matches the docx 111-twip spacing-before
FONT_PT = 10.0
LINE_SPACING = 1.18   # multiple of font height between baselines

# --- Fields -------------------------------------------------------------
# The bill values a label field can pull, in the default order, with the
# human-readable name shown in the field editor.
FIELD_VARS = ["serial", "series", "pattern", "note", "catalog", "position",
              "denomination", "front_plate", "back_plate"]
FIELD_LABELS = {
    "serial": "Serial",
    "series": "Series",
    "pattern": "Pattern",
    "note": "Note",
    "catalog": "Catalog",
    "position": "Position",
    "denomination": "Denomination",
    "front_plate": "Front Plate",
    "back_plate": "Back Plate",
}


@dataclass
class LabelField:
    var: str
    caption: str = ""
    enabled: bool = True
    same_line: bool = False
    # Per-field font overrides (blank family / 0 size = inherit the base 10pt).
    font_family: str = ""
    font_size: float = 0.0
    bold: bool = False
    italic: bool = False
    # Layout: horizontal alignment of the line this field starts (left/center/
    # right; a same-line field follows its line's starter), and whether to leave a
    # blank line above it for vertical spacing.
    align: str = "left"
    space_before: bool = False

    def has_font_override(self) -> bool:
        return bool(self.font_family or self.font_size or self.bold or self.italic)

    def to_dict(self) -> dict:
        return {"var": self.var, "caption": self.caption,
                "enabled": self.enabled, "same_line": self.same_line,
                "font_family": self.font_family, "font_size": self.font_size,
                "bold": self.bold, "italic": self.italic,
                "align": self.align, "space_before": self.space_before}


@dataclass
class LabelTemplate:
    """A label profile: physical size + base font + an ordered list of fields."""
    fields: List[LabelField]
    width_in: float = LABEL_W_IN
    height_in: float = LABEL_H_IN
    base_font_size: float = FONT_PT   # points; fields inherit this unless overridden
    base_font_family: str = ""        # family all fields inherit unless overridden ("" = system default)
    valign: str = "top"               # vertical placement of the text block: top/middle/bottom
    margin_in: float = MARGIN_IN      # blank border (inches) on all sides

    @staticmethod
    def default() -> "LabelTemplate":
        return LabelTemplate([
            LabelField("serial",   "",        True,  False),
            LabelField("series",   "SERIES ", True,  True),
            LabelField("pattern",  "",        True,  False),
            LabelField("note",     "",        True,  False),
            LabelField("catalog",  "Cat: ",   False, False),
            LabelField("position", "Pos: ",   False, True),
            LabelField("denomination", "",    False, False),
            LabelField("front_plate",  "FP ", False, False),
            LabelField("back_plate",   "BP ", False, True),
        ])

    @staticmethod
    def from_dict(d: Optional[dict]) -> "LabelTemplate":
        """Build a template from stored settings, tolerating missing/unknown."""
        if not d or not isinstance(d, dict):
            return LabelTemplate.default()
        defaults = {f.var: f for f in LabelTemplate.default().fields}
        seen = set()
        fields: List[LabelField] = []
        for entry in d.get("fields", []):
            if not isinstance(entry, dict):
                continue
            var = entry.get("var")
            if var not in FIELD_LABELS or var in seen:
                continue
            seen.add(var)
            dflt = defaults[var]
            fields.append(LabelField(
                var=var,
                caption=entry.get("caption", dflt.caption),
                enabled=bool(entry.get("enabled", dflt.enabled)),
                same_line=bool(entry.get("same_line", dflt.same_line)),
                font_family=entry.get("font_family", ""),
                font_size=float(entry.get("font_size", 0) or 0),
                bold=bool(entry.get("bold", False)),
                italic=bool(entry.get("italic", False)),
                align=entry.get("align", "left") or "left",
                space_before=bool(entry.get("space_before", False)),
            ))
        # Append any fields missing from the stored template (disabled), so new
        # field types show up in the editor after an upgrade.
        for var in FIELD_VARS:
            if var not in seen:
                f = defaults[var]
                fields.append(LabelField(f.var, f.caption, False, f.same_line))
        try:
            w = float(d.get("width_in", LABEL_W_IN)) or LABEL_W_IN
            h = float(d.get("height_in", LABEL_H_IN)) or LABEL_H_IN
            base = float(d.get("base_font_size", FONT_PT)) or FONT_PT
            margin = float(d.get("margin_in", MARGIN_IN))
        except (TypeError, ValueError):
            w, h, base, margin = LABEL_W_IN, LABEL_H_IN, FONT_PT, MARGIN_IN
        # Negative margins are allowed (pull text toward / past the edge).
        margin = max(-0.5, min(1.0, margin))
        return LabelTemplate(fields, width_in=w, height_in=h, base_font_size=base,
                             base_font_family=d.get("base_font_family", "") or "",
                             valign=d.get("valign", "top") or "top", margin_in=margin)

    def to_dict(self) -> dict:
        return {"fields": [f.to_dict() for f in self.fields],
                "width_in": self.width_in, "height_in": self.height_in,
                "base_font_size": self.base_font_size,
                "base_font_family": self.base_font_family,
                "valign": self.valign, "margin_in": self.margin_in}

    def render_paragraphs(self, values: Dict[str, str]):
        """Render the label as paragraphs of styled runs from a {var: value} map.

        Returns a list of paragraphs; each paragraph is a list of
        (text, LabelField|None) runs -- the field carries the per-field font, and
        None marks a plain (base-font) separator space between same-line fields.
        Fields with an empty value are skipped (no blank lines). same_line fields
        continue the previous paragraph.
        """
        paras: List[List[Tuple[str, Optional["LabelField"]]]] = []
        for f in self.fields:
            if not f.enabled:
                continue
            val = values.get(f.var, "")
            val = "" if val is None else str(val).strip()
            if not val:
                continue
            piece = (f.caption or "") + val
            if f.same_line and paras:
                if not (f.caption and f.caption[:1].isspace()):
                    paras[-1].append((" ", None))
                paras[-1].append((piece, f))
            else:
                paras.append([(piece, f)])
        return paras

    def render(self, values: Dict[str, str]) -> List[str]:
        """Plain-text lines (per paragraph), for callers that don't need fonts."""
        return ["".join(text for text, _ in para)
                for para in self.render_paragraphs(values)]


@dataclass
class LabelData:
    """The content of one printed label.

    Field values come from the bill; `lines` is the per-bill freeform override
    (prints verbatim when not None). `front_file` lets the caller persist an edit
    to the right bill; the renderer ignores it.
    """
    serial: str = ""
    series: str = ""
    pattern: str = ""
    note: str = ""
    catalog: str = ""
    position: str = ""
    denomination: str = ""
    front_plate: str = ""
    back_plate: str = ""
    front_file: str = ""
    suggested_note: str = ""   # auto-derived line-3 suggestion (not rendered directly)
    lines: Optional[List[str]] = None

    def values(self) -> Dict[str, str]:
        return {
            "serial": self.serial,
            "series": self.series,
            "pattern": self.pattern,
            "note": self.note,
            "catalog": self.catalog,
            "position": str(self.position) if self.position else "",
            "denomination": self.denomination,
            "front_plate": self.front_plate,
            "back_plate": self.back_plate,
        }

    def template_lines(self, template: LabelTemplate) -> List[str]:
        """The lines this bill produces from a template (ignores any override)."""
        return template.render(self.values())


def label_lines(data: LabelData, template: LabelTemplate,
                uppercase: bool = False) -> List[str]:
    """The logical lines for a label: the per-bill override if set, else the
    template render."""
    lines = list(data.lines) if data.lines is not None else data.template_lines(template)
    if uppercase:
        lines = [ln.upper() for ln in lines]
    return lines


def label_paragraphs(data: LabelData, template: LabelTemplate, uppercase: bool = False):
    """Paragraphs of styled runs for a label (override -> plain base-font runs)."""
    if data.lines is not None:
        src = [ln.upper() for ln in data.lines] if uppercase else list(data.lines)
        return [[(ln, None)] for ln in src]
    paras = template.render_paragraphs(data.values())
    if uppercase:
        paras = [[(text.upper(), fld) for text, fld in para] for para in paras]
    return paras


def _base_font(dpi: float, base_pt: float = FONT_PT, base_family: str = "") -> QFont:
    f = QFont()
    if base_family:
        f.setFamily(base_family)
    f.setPixelSize(max(1, round(base_pt * dpi / 72.0)))
    return f


def _field_font(dpi: float, fld: Optional[LabelField], base_pt: float = FONT_PT,
                base_family: str = "") -> QFont:
    """The font for a run: the profile's base family/size, overlaid with the
    field's own overrides (family/size/bold/italic)."""
    f = QFont()
    family = (fld.font_family if (fld and fld.font_family) else base_family)
    if family:
        f.setFamily(family)
    size_pt = (fld.font_size if (fld and fld.font_size) else base_pt)
    f.setPixelSize(max(1, round(size_pt * dpi / 72.0)))
    if fld is not None:
        f.setBold(bool(fld.bold))
        f.setItalic(bool(fld.italic))
    return f


# A laid-out token: text, its QFont, its metrics, and advance width.
_Tok = Tuple[str, QFont, QFontMetricsF, float]


def _line_starter(para):
    """The first field of a paragraph (drives its spacing)."""
    for _, fld in para:
        if fld is not None:
            return fld
    return None


def _strip(toks: List[_Tok]) -> List[_Tok]:
    """Drop leading/trailing whitespace tokens (they skew width/alignment)."""
    line = list(toks)
    while line and line[0][0].isspace():
        line.pop(0)
    while line and line[-1][0].isspace():
        line.pop()
    return line


def _wrap_tokens(toks: List[_Tok], avail_w: float) -> List[List[_Tok]]:
    """Greedy word-wrap tokens to `avail_w`, hard-breaking any too-long word."""
    visual: List[List[_Tok]] = []
    cur: List[_Tok] = []
    cur_w = 0.0
    for tok, font, fm, w in toks:
        is_space = tok.isspace()
        if not is_space and cur and cur_w + w > avail_w:
            visual.append(cur)
            cur, cur_w = [], 0.0
        if not is_space and not cur and w > avail_w:
            piece, pw = "", 0.0
            for ch in tok:
                cw = fm.horizontalAdvance(ch)
                if piece and pw + cw > avail_w:
                    visual.append([(piece, font, fm, pw)])
                    piece, pw = ch, cw
                else:
                    piece += ch
                    pw += cw
            if piece:
                cur, cur_w = [(piece, font, fm, pw)], pw
            continue
        cur.append((tok, font, fm, w))
        cur_w += w
    visual.append(cur)  # last line (may be empty -> an empty paragraph)
    return visual


def _layout(dpi: float, data: LabelData, template: LabelTemplate,
            uppercase: bool, avail_w: float, margin: float):
    """Lay the label out into fully-positioned visual lines.

    Own the layout (rather than QTextDocument) so mixed font sizes on one line get
    a correct line height (max of the runs). Horizontal placement is resolved here
    too: a line with a single alignment is left/center/right as a whole, while a
    line whose fields use *different* alignments is split into left / center /
    right segments (so e.g. Serial-left + Series-right spread to the edges).

    Returns (lines, total_height) where each line is (placed, ascent, line_height)
    and placed is a list of (text, font, x) with x an absolute label coordinate.
    """
    base_pt = template.base_font_size or FONT_PT
    base_family = template.base_font_family or ""
    base_fm = QFontMetricsF(_base_font(dpi, base_pt, base_family))
    base_asc, base_lh = base_fm.ascent(), base_fm.height() * LINE_SPACING

    lines = []      # (placed, ascent, line_height)
    total_h = 0.0

    def tokenize(runs):
        toks: List[_Tok] = []
        for text, fld in runs:
            font = _field_font(dpi, fld, base_pt, base_family)
            fm = QFontMetricsF(font)
            for tok in re.findall(r"\s+|\S+", text):
                toks.append((tok, font, fm, fm.horizontalAdvance(tok)))
        return toks

    def metrics(toks):
        asc = max((t[2].ascent() for t in toks), default=base_asc)
        lh = max((t[2].height() for t in toks), default=base_fm.height()) * LINE_SPACING
        return asc, lh

    def place(toks, x0):
        out, x = [], x0
        for tok, font, fm, w in toks:
            out.append((tok, font, x))
            x += w
        return out

    def emit(placed, asc, lh):
        nonlocal total_h
        lines.append((placed, asc, lh))
        total_h += lh

    for para in label_paragraphs(data, template, uppercase):
        starter = _line_starter(para)
        if starter and starter.space_before:
            emit([], base_asc, base_lh)  # blank spacer line

        aligns = {(fld.align if fld else "left") for _, fld in para}
        if len(aligns) > 1:
            # Mixed alignments on one line -> left / center / right segments.
            segs = {"left": [], "center": [], "right": []}
            for text, fld in para:
                segs[(fld.align if fld else "left")].append((text, fld))
            L = _strip(tokenize(segs["left"]))
            C = _strip(tokenize(segs["center"]))
            R = _strip(tokenize(segs["right"]))
            asc, lh = metrics(L + C + R)
            Cw = sum(t[3] for t in C)
            Rw = sum(t[3] for t in R)
            placed = []
            placed += place(L, margin)
            placed += place(C, margin + max(0.0, (avail_w - Cw) / 2.0))
            placed += place(R, margin + max(0.0, avail_w - Rw))
            emit(placed, asc, lh)
        else:
            align = aligns.pop() if aligns else "left"
            for vline in _wrap_tokens(tokenize(para), avail_w):
                vt = _strip(vline)
                asc, lh = metrics(vt)
                line_w = sum(t[3] for t in vt)
                if align == "center":
                    x0 = margin + max(0.0, (avail_w - line_w) / 2.0)
                elif align == "right":
                    x0 = margin + max(0.0, avail_w - line_w)
                else:
                    x0 = margin
                emit(place(vt, x0), asc, lh)

    return lines, total_h


def draw_label(painter: QPainter, dpi: float, data: LabelData,
               template: LabelTemplate, uppercase: bool = False,
               draw_background: bool = True) -> bool:
    """Draw one label into `painter` (device pixels at `dpi`), top-left at (0,0).

    Returns True if the text overflows the label height. Overflowing content is
    clipped, matching what the physical label would show.
    """
    w = template.width_in * dpi
    h = template.height_in * dpi
    margin = template.margin_in * dpi   # may be negative (pull text past the edge)

    if draw_background:
        painter.fillRect(0, 0, int(round(w)), int(round(h)), QColor("white"))

    avail_w = max(1.0, w - 2 * margin)
    lines, total_h = _layout(dpi, data, template, uppercase, avail_w, margin)

    usable_h = h - margin
    # Vertical placement of the whole text block within the label.
    extra = usable_h - total_h
    valign = (template.valign or "top")
    if valign == "middle":
        y = margin + max(0.0, extra / 2.0)
    elif valign == "bottom":
        y = margin + max(0.0, extra)
    else:
        y = margin

    painter.save()
    painter.setClipRect(0, 0, int(round(w)), int(round(h)))
    painter.setPen(QColor("black"))
    for placed, asc, line_h in lines:
        baseline = y + asc
        for tok, font, x in placed:
            painter.setFont(font)
            painter.drawText(int(round(x)), int(round(baseline)), tok)
        y += line_h
    painter.restore()

    return total_h > usable_h + 1.0


def render_image(data: LabelData, template: LabelTemplate,
                 uppercase: bool = False, dpi: float = 200.0):
    """Render one label to a QImage for on-screen preview.

    Returns (image, overflow_bool) using the same `draw_label` path as exports.
    """
    img = QImage(round(template.width_in * dpi), round(template.height_in * dpi),
                 QImage.Format_ARGB32)
    img.fill(Qt.white)
    painter = QPainter(img)
    try:
        overflow = draw_label(painter, dpi, data, template, uppercase)
    finally:
        painter.end()
    return img, overflow


def export_pdf(path: str, items: List[LabelData], template: LabelTemplate,
               uppercase: bool = False, dpi: int = 300) -> int:
    """Write one label per item (at the template's size) to a multi-page PDF."""
    writer = QPdfWriter(path)
    writer.setResolution(dpi)
    writer.setPageSize(QPageSize(QSizeF(template.width_in, template.height_in), QPageSize.Inch))
    writer.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Inch)

    painter = QPainter(writer)
    try:
        for i, item in enumerate(items):
            if i > 0:
                writer.newPage()
            draw_label(painter, dpi, item, template, uppercase, draw_background=False)
    finally:
        painter.end()
    return len(items)


def export_docx(path: str, items: List[LabelData], template: LabelTemplate,
                uppercase: bool = False) -> int:
    """Write the labels to an editable .docx (2x1 inch labels, one per section).

    Word wraps long lines within the cell on its own, so we emit one paragraph
    per logical line. Raises ImportError if python-docx isn't installed.
    """
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.oxml.ns import qn

    doc = Document()
    first = True
    for item in items:
        if first:
            section = doc.sections[0]
            first = False
        else:
            section = doc.add_section()
        section.page_width = Inches(template.width_in)
        section.page_height = Inches(template.height_in)
        section.left_margin = Inches(0)
        section.right_margin = Inches(0)
        section.top_margin = Inches(0)
        section.bottom_margin = Inches(0)

        table = doc.add_table(rows=1, cols=1)
        table.autofit = False
        tbl_pr = table._element.find(qn('w:tblPr'))
        borders = tbl_pr.makeelement(qn('w:tblBorders'), {})
        for bn in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
            borders.append(borders.makeelement(qn(f'w:{bn}'), {
                qn('w:val'): 'none', qn('w:sz'): '0',
                qn('w:space'): '0', qn('w:color'): 'auto'}))
        tbl_pr.append(borders)
        grid = table._element.find(qn('w:tblGrid'))
        if grid is not None:
            for col in grid.findall(qn('w:gridCol')):
                col.set(qn('w:w'), str(int(template.width_in * 1440)))

        # Vertical alignment of the cell contents (top/middle/bottom).
        cell = table.cell(0, 0)
        tc_pr = cell._element.find(qn('w:tcPr'))
        if tc_pr is None:
            tc_pr = cell._element.makeelement(qn('w:tcPr'), {})
            cell._element.insert(0, tc_pr)
        v = {"middle": "center", "bottom": "bottom"}.get(template.valign or "top", "top")
        tc_pr.append(tc_pr.makeelement(qn('w:vAlign'), {qn('w:val'): v}))

        base_pt = template.base_font_size or FONT_PT
        base_family = template.base_font_family or ""
        _JC = {"center": "center", "right": "right"}
        # Word can't cleanly indent past the cell edge, so clamp its margin to >=0
        # (the PDF/preview honor negative margins; Word is the editable fallback).
        margin_tw = str(int(max(0.0, template.margin_in) * 1440))  # inches -> twips
        paras = label_paragraphs(item, template, uppercase) or [[("", None)]]

        # Reuse the cell's initial empty paragraph first, then add more.
        _state = {"used_first": False, "top_pad": False}

        def next_para():
            if not _state["used_first"]:
                _state["used_first"] = True
                p = cell.paragraphs[0]
            else:
                p = cell.add_paragraph()
            ppr = p._element.find(qn('w:pPr'))
            if ppr is None:
                ppr = p._element.makeelement(qn('w:pPr'), {})
                p._element.insert(0, ppr)
            if not _state["top_pad"]:
                _state["top_pad"] = True
                ppr.append(ppr.makeelement(qn('w:spacing'), {qn('w:before'): margin_tw}))
            ppr.append(ppr.makeelement(qn('w:ind'), {qn('w:left'): margin_tw, qn('w:right'): margin_tw}))
            return p, ppr

        for para in paras:
            starter = _line_starter(para)
            if starter and starter.space_before:
                next_para()  # blank spacer line
            p, ppr = next_para()
            jc = _JC.get(starter.align if starter else "left")
            if jc:
                ppr.append(ppr.makeelement(qn('w:jc'), {qn('w:val'): jc}))
            for text, fld in para:
                run = p.add_run(text)
                run.font.size = Pt(fld.font_size if (fld and fld.font_size) else base_pt)
                family = (fld.font_family if (fld and fld.font_family) else base_family)
                if family:
                    run.font.name = family
                if fld is not None:
                    run.font.bold = bool(fld.bold)
                    run.font.italic = bool(fld.italic)

    body = doc.element.body
    first_p = body.find(qn('w:p'))
    first_tbl = body.find(qn('w:tbl'))
    if first_p is not None and first_tbl is not None:
        if list(body).index(first_p) < list(body).index(first_tbl):
            body.remove(first_p)

    doc.save(path)
    return len(items)
