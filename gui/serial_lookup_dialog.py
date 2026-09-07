"""Serial Lookup dialog.

Type a serial number and see which patterns it matches, with a stylized overlay
preview of the selected match. This is Phase 1 of the manual-serial feature --
lookup only; injecting a synthetic result into a processed run ("add to run") is
a planned later phase.

No bill image or OCR is involved: it reuses the pattern engine's classify() and
the DigitPreviewWidget stylized renderer (an obvious graphic, not a photo), so it
is instant and works with no run loaded. Patterns that need image-derived
metadata (GAS_PUMP, SEAL_SHIFT, plate/mule) simply won't match a typed serial --
those are image findings, not serial findings.
"""
import re

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QListWidget,
    QListWidgetItem, QWidget, QSplitter,
)

from .pattern_dialog import DigitPreviewWidget


class SerialLookupDialog(QDialog):
    """Non-modal utility: serial in -> matched patterns + overlay preview out."""

    def __init__(self, pattern_engine, parent=None, initial_serial: str = ""):
        super().__init__(parent)
        self.engine = pattern_engine
        self._matches = []
        self.setWindowTitle("Serial Lookup")
        self.resize(760, 500)
        self._build_ui()
        if initial_serial:
            self.serial_edit.setText(initial_serial)
        self.serial_edit.setFocus()
        self._on_changed(self.serial_edit.text())

    def _build_ui(self):
        root = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Serial:"))
        self.serial_edit = QLineEdit()
        self.serial_edit.setPlaceholderText("A12345678B   or   12345678")
        mono = QFont("Consolas, Monaco, monospace")
        mono.setPointSize(14)
        self.serial_edit.setFont(mono)
        self.serial_edit.textChanged.connect(self._on_changed)
        row.addWidget(self.serial_edit, 1)
        root.addLayout(row)

        self.summary = QLabel("")
        self.summary.setStyleSheet("color:#555;")
        root.addWidget(self.summary)

        # Stylized serial graphic (tan strip + green digits + overlay), NOT a photo.
        self.preview = DigitPreviewWidget()
        self.preview.setMinimumHeight(135)
        root.addWidget(self.preview)

        split = QSplitter(Qt.Horizontal)
        self.matches_list = QListWidget()
        self.matches_list.currentRowChanged.connect(self._on_select)
        split.addWidget(self.matches_list)

        self.details = QLabel("")
        self.details.setWordWrap(True)
        self.details.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.details.setStyleSheet("padding:6px;")
        dwrap = QWidget()
        dlay = QVBoxLayout(dwrap)
        dlay.setContentsMargins(0, 0, 0, 0)
        dlay.addWidget(self.details, 1)
        split.addWidget(dwrap)
        split.setSizes([280, 460])
        root.addWidget(split, 1)

    @staticmethod
    def _normalize(text: str):
        """Return (clean_serial, digits_only). Strips non-alphanumerics, uppercases."""
        s = re.sub(r'[^A-Za-z0-9]', '', text).upper()
        digits = re.sub(r'[^0-9]', '', s)
        return s, digits

    def _on_changed(self, text: str):
        s, digits = self._normalize(text)
        self.matches_list.clear()
        self._matches = []
        self.details.setText("")
        self.preview.set_highlights([], [])
        self.preview.set_group_boxes([])

        if len(digits) != 8:
            self.summary.setText(
                "Enter 8 digits (optionally with prefix/suffix letters, e.g. A12345678B).")
            self.preview.set_serial(s or "--------")
            return

        self.preview.set_serial(s)
        try:
            metadata = {"gas_pump_threshold": self.engine.get_gas_pump_threshold()}
            matches = self.engine.classify(s, metadata)
        except Exception as e:  # pragma: no cover - defensive
            self.summary.setText(f"Lookup error: {e}")
            return

        self._matches = matches
        if not matches:
            self.summary.setText("No fancy patterns matched.")
            return

        self.summary.setText(f"{len(matches)} pattern(s) matched — click one to see its overlay.")
        for m in matches:
            info = self.engine.get_pattern_info(m.name) or {}
            disp = info.get('display_name') or m.name
            lib = info.get('library', '')
            self.matches_list.addItem(QListWidgetItem(f"{disp}  ({lib})" if lib else disp))
        self.matches_list.setCurrentRow(0)

    def _on_select(self, idx: int):
        if not (0 <= idx < len(self._matches)):
            self.preview.set_highlights([], [])
            self.preview.set_group_boxes([])
            return
        m = self._matches[idx]
        self.preview.set_highlights(m.highlights, m.connectors)
        self.preview.set_group_boxes(m.group_boxes)

        info = self.engine.get_pattern_info(m.name) or {}
        lines = [f"<b>{info.get('display_name') or m.name}</b>"]
        if info.get('library'):
            lines.append(f"Library: {info['library']}")
        if info.get('odds'):
            lines.append(f"Rarity: {info['odds']}")
        price = info.get('price') or info.get('price_range')
        if price:
            lines.append(f"Est. Price: {price}")
        if getattr(m, 'message', ''):
            lines.append(f"<i>{m.message}</i>")
        self.details.setText("<br>".join(lines))
