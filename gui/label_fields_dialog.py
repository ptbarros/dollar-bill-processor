"""Label Fields editor — edit the 2x1 label TEMPLATE.

A label is composed of fields (Serial, Series, Pattern, Note, Catalog,
Position). This dialog edits the template that all labels share: which fields
show, the caption before each (change "SERIES " to "Ser: " once, everywhere),
whether a field continues the previous line or starts a new one, and the order.

A live sample label at the top updates as you edit, so you can see the effect
before applying it to every label. Saved as a named profile (settings.label_profiles).
"""
from dataclasses import replace
from typing import List

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QCheckBox,
    QPushButton, QWidget, QScrollArea, QFrame, QDialogButtonBox, QFontDialog,
    QDoubleSpinBox, QSpinBox, QFontComboBox, QStyle, QComboBox,
)

from .label_render import (
    LabelTemplate, LabelField, LabelData, FIELD_LABELS, render_image, FONT_PT,
)

_THUMB_W = 280
_THUMB_H = 140


class _FieldRow(QFrame):
    """One field: enable + name + caption + same-line + reorder."""

    def __init__(self, field: LabelField, on_change, on_move, parent=None):
        super().__init__(parent)
        self.field = field
        self._on_change = on_change
        self._on_move = on_move
        self.setFrameShape(QFrame.StyledPanel)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(3)
        style = self.style()

        # Top row: enable, name, caption, same-line, alignment.
        top = QHBoxLayout()
        top.setSpacing(8)
        self.enable_chk = QCheckBox()
        self.enable_chk.setChecked(field.enabled)
        self.enable_chk.setToolTip("Show this field on the label")
        self.enable_chk.toggled.connect(self._enabled_changed)
        top.addWidget(self.enable_chk)

        name = QLabel(FIELD_LABELS.get(field.var, field.var))
        name.setFixedWidth(90)
        name.setStyleSheet("font-weight: bold;")
        top.addWidget(name)

        top.addWidget(QLabel("caption:"))
        self.caption_edit = QLineEdit(field.caption)
        self.caption_edit.setFixedWidth(110)
        self.caption_edit.setToolTip("Text printed before the value, e.g. \"Ser: \"")
        self.caption_edit.textChanged.connect(self._caption_changed)
        top.addWidget(self.caption_edit)

        self.same_chk = QCheckBox("same line")
        self.same_chk.setChecked(field.same_line)
        self.same_chk.setToolTip("Continue the previous line instead of starting a new one")
        self.same_chk.toggled.connect(self._same_changed)
        top.addWidget(self.same_chk)

        top.addWidget(QLabel("align:"))
        self.align_combo = QComboBox()
        self.align_combo.addItems(["Left", "Center", "Right"])
        self.align_combo.setCurrentIndex({"left": 0, "center": 1, "right": 2}.get(field.align, 0))
        self.align_combo.setToolTip("Horizontal position of this field's line")
        self.align_combo.currentIndexChanged.connect(self._align_changed)
        top.addWidget(self.align_combo)
        top.addStretch(1)
        outer.addLayout(top)

        # Bottom row: font controls, spacer toggle, reorder.
        bot = QHBoxLayout()
        bot.setSpacing(8)
        bot.addSpacing(114)
        self.font_btn = QPushButton("Font…")
        self.font_btn.setToolTip("Set this field's font, size, bold, italic")
        self.font_btn.clicked.connect(self._pick_font)
        bot.addWidget(self.font_btn)
        self.font_lbl = QLabel("")
        self.font_lbl.setFixedWidth(110)
        self.font_lbl.setStyleSheet("color: #555;")
        bot.addWidget(self.font_lbl)
        clear_font = QPushButton("Clear font")
        clear_font.setToolTip("Use the profile's base font for this field")
        clear_font.clicked.connect(self._clear_font)
        bot.addWidget(clear_font)
        self._update_font_label()

        self.space_chk = QCheckBox("space above")
        self.space_chk.setChecked(field.space_before)
        self.space_chk.setToolTip("Leave a blank line above this field for spacing")
        self.space_chk.toggled.connect(self._space_changed)
        bot.addWidget(self.space_chk)

        bot.addStretch(1)
        # Reorder buttons drawn with the style's arrow icons (font glyphs like
        # ↑/↓ render as blank squares on some Linux setups).
        up = QPushButton("Up")
        up.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_ArrowUp))
        up.setToolTip("Move this field up")
        up.clicked.connect(lambda: self._on_move(self, -1))
        bot.addWidget(up)
        down = QPushButton("Down")
        down.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_ArrowDown))
        down.setToolTip("Move this field down")
        down.clicked.connect(lambda: self._on_move(self, +1))
        bot.addWidget(down)
        outer.addLayout(bot)

    def _enabled_changed(self, v):
        self.field.enabled = v
        self._on_change()

    def _caption_changed(self, text):
        self.field.caption = text
        self._on_change()

    def _same_changed(self, v):
        self.field.same_line = v
        self._on_change()

    def _align_changed(self, idx):
        self.field.align = ("left", "center", "right")[idx]
        self._on_change()

    def _space_changed(self, v):
        self.field.space_before = v
        self._on_change()

    def _update_font_label(self):
        f = self.field
        if not f.has_font_override():
            self.font_lbl.setText("(default)")
            return
        parts = [f.font_family or "default", f"{int(f.font_size or FONT_PT)}pt"]
        if f.bold:
            parts.append("B")
        if f.italic:
            parts.append("I")
        self.font_lbl.setText(" ".join(parts))

    def _pick_font(self):
        # Seed the picker with the field's current effective font.
        initial = QFont(self.field.font_family or "")
        initial.setPointSizeF(self.field.font_size or FONT_PT)
        initial.setBold(self.field.bold)
        initial.setItalic(self.field.italic)
        res = QFontDialog.getFont(initial, self, "Field Font")
        # PySide/PyQt differ on tuple order; detect the QFont element.
        a, b = res
        font, ok = (a, b) if isinstance(a, QFont) else (b, a)
        if not ok:
            return
        self.field.font_family = font.family()
        self.field.font_size = float(font.pointSizeF()) if font.pointSizeF() > 0 else float(font.pixelSize())
        self.field.bold = font.bold()
        self.field.italic = font.italic()
        self._update_font_label()
        self._on_change()

    def _clear_font(self):
        self.field.font_family = ""
        self.field.font_size = 0.0
        self.field.bold = False
        self.field.italic = False
        self._update_font_label()
        self._on_change()


class LabelFieldsDialog(QDialog):
    """Edit the label template; returns the new template via `result_template`."""

    def __init__(self, template: LabelTemplate, sample: LabelData,
                 uppercase: bool = False, profile_name: str = "", parent=None):
        super().__init__(parent)
        # Work on a copy so Cancel leaves the original untouched.
        self._fields: List[LabelField] = [replace(f) for f in template.fields]
        self._width = template.width_in
        self._height = template.height_in
        self._base = template.base_font_size
        self._base_family = template.base_font_family or ""
        self._valign = template.valign or "top"
        self._margin = template.margin_in
        self._sample = sample
        self._uppercase = uppercase
        self.result_template = None
        self.setWindowTitle(f"Edit Label Profile — {profile_name}" if profile_name
                            else "Edit Label Profile")
        self.resize(940, 660)
        self._build_ui()
        self._refresh_preview()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Label size + base font (defines the profile's physical label).
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Label size (in):"))
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.5, 12.0)
        self.width_spin.setSingleStep(0.25)
        self.width_spin.setDecimals(2)
        self.width_spin.setValue(self._width)
        self.width_spin.valueChanged.connect(self._size_changed)
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QLabel("×"))
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.5, 12.0)
        self.height_spin.setSingleStep(0.25)
        self.height_spin.setDecimals(2)
        self.height_spin.setValue(self._height)
        self.height_spin.valueChanged.connect(self._size_changed)
        size_row.addWidget(self.height_spin)
        size_row.addSpacing(16)
        size_row.addWidget(QLabel("Base font:"))
        self.base_font_combo = QFontComboBox()
        self.base_font_combo.setToolTip("Default font family for all fields (each field "
                                        "can still override its own)")
        self.base_font_combo.setMaximumWidth(150)
        self.base_font_combo.blockSignals(True)
        if self._base_family:
            self.base_font_combo.setCurrentFont(QFont(self._base_family))
        self.base_font_combo.blockSignals(False)
        # Only user changes set the family, so an untouched profile keeps "" (=
        # the system default) rather than baking in this machine's default font.
        self.base_font_combo.currentFontChanged.connect(self._base_family_changed)
        size_row.addWidget(self.base_font_combo)
        size_row.addWidget(QLabel("pt:"))
        self.base_spin = QSpinBox()
        self.base_spin.setRange(5, 72)
        self.base_spin.setValue(int(self._base))
        self.base_spin.setToolTip("Default size for fields without their own font")
        self.base_spin.valueChanged.connect(self._size_changed)
        size_row.addWidget(self.base_spin)
        size_row.addSpacing(16)
        size_row.addWidget(QLabel("Margin (in):"))
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(-0.5, 1.0)
        self.margin_spin.setSingleStep(0.02)
        self.margin_spin.setDecimals(2)
        self.margin_spin.setValue(self._margin)
        self.margin_spin.setToolTip("Blank border on all sides. Negative pulls the "
                                    "text toward (or past) the label edge.")
        self.margin_spin.valueChanged.connect(self._size_changed)
        size_row.addWidget(self.margin_spin)
        size_row.addSpacing(16)
        size_row.addWidget(QLabel("Vertical:"))
        self.valign_combo = QComboBox()
        self.valign_combo.addItems(["Top", "Middle", "Bottom"])
        self.valign_combo.setCurrentIndex({"top": 0, "middle": 1, "bottom": 2}.get(self._valign, 0))
        self.valign_combo.setToolTip("Vertical placement of the whole text block on the label")
        self.valign_combo.currentIndexChanged.connect(self._size_changed)
        size_row.addWidget(self.valign_combo)
        size_row.addStretch(1)
        layout.addLayout(size_row)

        # Live sample label.
        self.preview = QLabel()
        self.preview.setFixedSize(_THUMB_W, _THUMB_H)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("border: 1px solid #999; background: white;")
        pv_row = QHBoxLayout()
        pv_row.addStretch(1)
        pv_row.addWidget(self.preview)
        pv_row.addStretch(1)
        layout.addLayout(pv_row)

        hint = QLabel("Size, captions and layout here apply to every label on this "
                      "profile. Empty fields (e.g. a bill with no note) are skipped.")
        hint.setStyleSheet("color: #666;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # Field rows.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._container = QWidget()
        self._rows_layout = QVBoxLayout(self._container)
        self._rows_layout.setSpacing(4)
        self._scroll.setWidget(self._container)
        layout.addWidget(self._scroll, 1)
        self._rebuild_rows()

        # Buttons: restore defaults + OK/Cancel.
        btns = QHBoxLayout()
        reset = QPushButton("Restore Defaults")
        reset.clicked.connect(self._restore_defaults)
        btns.addWidget(reset)
        btns.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        btns.addWidget(bb)
        layout.addLayout(btns)

    def _rebuild_rows(self):
        # Clear existing row widgets.
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._row_widgets: List[_FieldRow] = []
        for f in self._fields:
            r = _FieldRow(f, self._refresh_preview, self._move_row, self._container)
            self._row_widgets.append(r)
            self._rows_layout.addWidget(r)
        self._rows_layout.addStretch(1)

    def _move_row(self, row: _FieldRow, delta: int):
        i = self._fields.index(row.field)
        j = i + delta
        if 0 <= j < len(self._fields):
            self._fields[i], self._fields[j] = self._fields[j], self._fields[i]
            self._rebuild_rows()
            self._refresh_preview()

    def _restore_defaults(self):
        # Reset the field layout only; keep the label size the user chose.
        self._fields = [replace(f) for f in LabelTemplate.default().fields]
        self._rebuild_rows()
        self._refresh_preview()

    def _size_changed(self):
        self._width = self.width_spin.value()
        self._height = self.height_spin.value()
        self._base = float(self.base_spin.value())
        self._margin = float(self.margin_spin.value())
        self._valign = ("top", "middle", "bottom")[self.valign_combo.currentIndex()]
        self._refresh_preview()

    def _base_family_changed(self, font: QFont):
        self._base_family = font.family()
        self._refresh_preview()

    def _template(self) -> LabelTemplate:
        return LabelTemplate(self._fields, width_in=self._width,
                             height_in=self._height, base_font_size=self._base,
                             base_font_family=self._base_family, valign=self._valign,
                             margin_in=self._margin)

    def _refresh_preview(self):
        img, _ = render_image(self._sample, self._template(), self._uppercase)
        self.preview.setPixmap(QPixmap.fromImage(img).scaled(
            _THUMB_W, _THUMB_H, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _accept(self):
        self.result_template = self._template()
        self.accept()
