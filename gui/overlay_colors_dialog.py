"""Overlay Colors tool.

Lets the user retune the overlay palette *slots* (blue, orange, magenta, ...)
that the contrast rotation draws from -- e.g. darken an orange that fades under
eyestrain, or lighten a blue that reads as black. Changes preview live in the
dialog and are applied + persisted only on OK (Cancel reverts).

The palette is a single source of truth in serial_overlay: overrides here feed
serial_overlay.set_overlay_overrides(), which rebuilds PATTERN_COLORS for BOTH
the drawn crops/main view and the pattern-preview widget.
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QColorDialog, QDialogButtonBox, QWidget,
)

import serial_overlay
from .pattern_dialog import DigitPreviewWidget


class OverlayColorsDialog(QDialog):
    def __init__(self, current_overrides: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Overlay Colors")
        self.setMinimumWidth(560)
        # Working copy of {slot: '#rrggbb'} overrides; committed only on accept.
        self._overrides = dict(current_overrides or {})
        self._swatches = {}
        self._build_ui()
        self._refresh_preview()

    # ---- UI ---------------------------------------------------------------- #
    def _build_ui(self):
        root = QVBoxLayout(self)

        intro = QLabel(
            "Retune the overlay palette. These are the color slots the overlay "
            "rotation uses; changing one updates every overlay that draws it "
            "(preview updates live; changes apply when you click OK).")
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#555;")
        root.addWidget(intro)

        # Live preview: a representative overlay exercising the rotation slots.
        self.preview = DigitPreviewWidget()
        self.preview.set_serial("A12345678B")
        # 7 distinct group names -> the 7 rotation slots (blue..hotpink); the 8th
        # digit shows a muted gray X (the excluded-digit style).
        highlights = [{'positions': [i], 'color': f'g{i}'} for i in range(7)]
        highlights.append({'positions': [7], 'color': 'gray', 'style': 'x'})
        # A couple of nested arcs so arc + stub colors show too.
        connectors = [{'from': 0, 'to': 6, 'color': 'g0'},
                      {'from': 1, 'to': 5, 'color': 'g1'}]
        self.preview.set_highlights(highlights, connectors)
        self.preview.set_group_boxes([])
        root.addWidget(self.preview)

        # One row per editable slot: name | swatch (click to pick) | Reset.
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        for r, slot in enumerate(serial_overlay.EDITABLE_SLOTS):
            grid.addWidget(QLabel(slot.capitalize()), r, 0)
            sw = QPushButton()
            sw.setFixedSize(120, 24)
            sw.setCursor(Qt.PointingHandCursor)
            sw.clicked.connect(lambda _c, s=slot: self._pick(s))
            grid.addWidget(sw, r, 1)
            self._swatches[slot] = sw
            reset = QPushButton("Reset")
            reset.clicked.connect(lambda _c, s=slot: self._reset_slot(s))
            grid.addWidget(reset, r, 2)
        gwrap = QWidget()
        gwrap.setLayout(grid)
        root.addWidget(gwrap)

        # Bottom buttons
        btm = QHBoxLayout()
        reset_all = QPushButton("Reset All")
        reset_all.clicked.connect(self._reset_all)
        btm.addWidget(reset_all)
        btm.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        btm.addWidget(buttons)
        root.addLayout(btm)

        for slot in serial_overlay.EDITABLE_SLOTS:
            self._update_swatch(slot)

    # ---- helpers ----------------------------------------------------------- #
    def _slot_hex(self, slot: str) -> str:
        return self._overrides.get(slot) or serial_overlay.default_slot_hex(slot)

    def _update_swatch(self, slot: str):
        h = self._slot_hex(slot)
        overridden = slot in self._overrides
        self._swatches[slot].setStyleSheet(
            f"background-color:{h}; border:1px solid #888;")
        self._swatches[slot].setText("" if not overridden else "●")
        self._swatches[slot].setToolTip(
            f"{h}" + ("  (customized)" if overridden else "  (default)"))

    def _refresh_preview(self):
        self.preview.set_palette_override(serial_overlay.build_palette(self._overrides))

    def _pick(self, slot: str):
        initial = QColor(self._slot_hex(slot))
        chosen = QColorDialog.getColor(initial, self, f"Pick color for '{slot}'")
        if chosen.isValid():
            self._overrides[slot] = chosen.name()  # '#rrggbb'
            self._update_swatch(slot)
            self._refresh_preview()

    def _reset_slot(self, slot: str):
        self._overrides.pop(slot, None)
        self._update_swatch(slot)
        self._refresh_preview()

    def _reset_all(self):
        self._overrides.clear()
        for slot in serial_overlay.EDITABLE_SLOTS:
            self._update_swatch(slot)
        self._refresh_preview()

    def result_overrides(self) -> dict:
        """The committed overrides (call after exec() returns Accepted)."""
        return dict(self._overrides)
