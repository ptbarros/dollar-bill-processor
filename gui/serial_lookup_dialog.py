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

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QImage, QPainter, QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QListWidget,
    QListWidgetItem, QWidget, QSplitter, QPushButton, QApplication,
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

        # Copy an image of the overlay + matched pattern name to the clipboard —
        # for pasting into a chat when explaining a pattern to the group.
        self.copy_btn = QPushButton("Copy Image")
        self.copy_btn.setToolTip("Copy a picture of this overlay and its pattern name "
                                 "to the clipboard (paste into a chat)")
        self.copy_btn.clicked.connect(self._copy_image)
        self.copy_btn.setEnabled(False)
        row.addWidget(self.copy_btn)
        root.addLayout(row)

        self.summary = QLabel("")
        self.summary.setStyleSheet("color:#555;")
        root.addWidget(self.summary)

        # Stylized serial graphic (tan strip + green digits + overlay), NOT a photo.
        self.preview = DigitPreviewWidget()
        self.preview.setMinimumHeight(135)
        root.addWidget(self.preview)

        self.splitter = QSplitter(Qt.Horizontal)
        self.matches_list = QListWidget()
        # Show full pattern names: don't elide, scroll if one is extremely long.
        self.matches_list.setTextElideMode(Qt.ElideNone)
        self.matches_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.matches_list.currentRowChanged.connect(self._on_select)
        self.splitter.addWidget(self.matches_list)

        self.details = QLabel("")
        self.details.setWordWrap(True)
        self.details.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.details.setStyleSheet("padding:6px;")
        dwrap = QWidget()
        dlay = QVBoxLayout(dwrap)
        dlay.setContentsMargins(0, 0, 0, 0)
        dlay.addWidget(self.details, 1)
        self.splitter.addWidget(dwrap)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([280, 460])
        root.addWidget(self.splitter, 1)

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
        self.copy_btn.setEnabled(False)

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
        self._fit_panes()

    def _fit_panes(self):
        """Size the match list to its longest name and widen the dialog if needed,
        so long pattern names + details fit without the user dragging the divider."""
        fm = self.matches_list.fontMetrics()
        text_w = max((fm.horizontalAdvance(self.matches_list.item(i).text())
                      for i in range(self.matches_list.count())), default=0)
        # + item padding + scrollbar allowance + frame; clamp to a sensible band.
        list_w = max(200, min(text_w + 44, 420))
        self.matches_list.setMinimumWidth(list_w)

        details_min = 340
        margins = 48
        want = list_w + details_min + margins
        # Only grow (never shrink under the user), and keep it on-screen.
        if self.width() < want:
            screen = self.screen().availableGeometry().width() if self.screen() else want
            self.resize(min(want, screen - 40), self.height())
        self.splitter.setSizes([list_w, max(details_min, self.width() - list_w - margins)])

    def _on_select(self, idx: int):
        if not (0 <= idx < len(self._matches)):
            self.preview.set_highlights([], [])
            self.preview.set_group_boxes([])
            self.copy_btn.setEnabled(False)
            return
        m = self._matches[idx]
        self.preview.set_highlights(m.highlights, m.connectors)
        self.preview.set_group_boxes(m.group_boxes)
        self.copy_btn.setEnabled(True)

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

    def _copy_image(self):
        """Compose the overlay preview + pattern name into one clipboard image."""
        idx = self.matches_list.currentRow()
        if not (0 <= idx < len(self._matches)):
            return
        m = self._matches[idx]
        info = self.engine.get_pattern_info(m.name) or {}
        name = info.get('display_name') or m.name
        serial = self._normalize(self.serial_edit.text())[0]

        # Grab exactly what the overlay widget is showing (serial strip + overlay).
        preview_pix = self.preview.grab()
        pw, ph = preview_pix.width(), preview_pix.height()

        pad = 16
        title_h = 52
        width = max(pw, 460) + pad * 2
        height = title_h + ph + pad * 2
        img = QImage(width, height, QImage.Format_ARGB32)
        img.fill(QColor("white"))
        p = QPainter(img)
        try:
            # Pattern name (bold) + serial / rarity subtitle.
            tf = QFont()
            tf.setPointSize(14)
            tf.setBold(True)
            p.setFont(tf)
            p.setPen(QColor("#1a1a1a"))
            p.drawText(pad, pad + 22, name)

            sf = QFont()
            sf.setPointSize(10)
            p.setFont(sf)
            p.setPen(QColor("#555555"))
            sub = serial
            if info.get('odds'):
                sub += f"    Odds: {info['odds']}"
            p.drawText(pad, pad + 44, sub)

            # Overlay preview, centered under the title.
            p.drawPixmap((width - pw) // 2, title_h + pad, preview_pix)
        finally:
            p.end()

        QApplication.clipboard().setImage(img)

        # Brief confirmation on the button.
        prev = self.copy_btn.text()
        self.copy_btn.setText("Copied ✓")
        QTimer.singleShot(1200, lambda: self.copy_btn.setText(prev))
