"""Read-only viewer for an imported review bundle (.zip).

Pages through the bills the FIL flagged with "Save for Review": the front/back
scans + serial crop on the left, the model's detection + his note on the right.
"""

from pathlib import Path
from typing import List

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QWidget, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap


class ReviewBundleDialog(QDialog):
    """Page through the bills in an imported review bundle (read-only)."""

    def __init__(self, items: List[dict], source_name: str = "", parent=None):
        super().__init__(parent)
        self._items = items or []
        self._idx = 0
        self.setWindowTitle(
            f"Review Bundle — {len(self._items)} bill(s)"
            + (f"  ({source_name})" if source_name else ""))
        self.resize(900, 640)
        self._build()
        self._show_current()

    def _build(self):
        outer = QVBoxLayout(self)

        body = QHBoxLayout()
        outer.addLayout(body, 1)

        # Left: stacked images in a scroll area.
        self._img_host = QWidget()
        self._img_layout = QVBoxLayout(self._img_host)
        self._img_layout.setAlignment(Qt.AlignTop)
        img_scroll = QScrollArea()
        img_scroll.setWidgetResizable(True)
        img_scroll.setWidget(self._img_host)
        img_scroll.setMinimumWidth(520)
        body.addWidget(img_scroll, 3)

        # Right: detected metadata + note.
        meta = QFrame()
        meta.setFrameShape(QFrame.StyledPanel)
        meta_l = QVBoxLayout(meta)
        meta_l.setAlignment(Qt.AlignTop)
        self._lbl_serial = self._field(meta_l, "Serial")
        self._lbl_patterns = self._field(meta_l, "Patterns")
        self._lbl_conf = self._field(meta_l, "Confidence")
        self._lbl_ts = self._field(meta_l, "Flagged")
        meta_l.addSpacing(8)
        note_cap = QLabel("Reviewer note")
        note_cap.setStyleSheet("color: gray; font-size: 11px;")
        meta_l.addWidget(note_cap)
        self._lbl_note = QLabel("")
        self._lbl_note.setWordWrap(True)
        self._lbl_note.setStyleSheet("font-size: 14px;")
        meta_l.addWidget(self._lbl_note)
        meta_l.addStretch(1)
        body.addWidget(meta, 2)

        # Bottom: pager.
        nav = QHBoxLayout()
        self._prev_btn = QPushButton("‹ Prev")
        self._prev_btn.clicked.connect(lambda: self._step(-1))
        self._next_btn = QPushButton("Next ›")
        self._next_btn.clicked.connect(lambda: self._step(1))
        self._counter = QLabel("")
        self._counter.setAlignment(Qt.AlignCenter)
        nav.addWidget(self._prev_btn)
        nav.addWidget(self._counter, 1)
        nav.addWidget(self._next_btn)
        outer.addLayout(nav)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        outer.addLayout(close_row)

    def _field(self, layout, caption: str) -> QLabel:
        cap = QLabel(caption)
        cap.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(cap)
        val = QLabel("")
        val.setWordWrap(True)
        val.setTextInteractionFlags(Qt.TextSelectableByMouse)
        val.setStyleSheet("font-size: 15px; font-weight: bold; margin-bottom: 6px;")
        layout.addWidget(val)
        return val

    def _step(self, d: int):
        if not self._items:
            return
        self._idx = max(0, min(len(self._items) - 1, self._idx + d))
        self._show_current()

    def _add_image(self, path: str, caption: str):
        if not path or not Path(path).exists():
            return
        pm = QPixmap(path)
        if pm.isNull():
            return
        cap = QLabel(caption)
        cap.setStyleSheet("color: gray; font-size: 11px;")
        self._img_layout.addWidget(cap)
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignLeft)
        # Scale down big scans to fit the panel width; never upscale.
        if pm.width() > 500:
            pm = pm.scaledToWidth(500, Qt.SmoothTransformation)
        lbl.setPixmap(pm)
        lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._img_layout.addWidget(lbl)

    def _clear_images(self):
        while self._img_layout.count():
            w = self._img_layout.takeAt(0).widget()
            if w is not None:
                w.deleteLater()

    def _show_current(self):
        if not self._items:
            self._counter.setText("No bills in this bundle")
            self._prev_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
            return
        it = self._items[self._idx]
        self._lbl_serial.setText(it.get("serial") or "—")
        self._lbl_patterns.setText(it.get("patterns") or "—")
        self._lbl_conf.setText(str(it.get("confidence") or "—"))
        self._lbl_ts.setText((it.get("timestamp") or "").replace("T", "  "))
        self._lbl_note.setText(it.get("note") or "(no note)")

        self._clear_images()
        self._add_image(it.get("serial_crop"), "Serial crop")
        self._add_image(it.get("front"), "Front scan")
        self._add_image(it.get("back"), "Back scan")

        self._counter.setText(f"Bill {self._idx + 1} of {len(self._items)}")
        self._prev_btn.setEnabled(self._idx > 0)
        self._next_btn.setEnabled(self._idx < len(self._items) - 1)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Left, Qt.Key_Right) and event.modifiers() == Qt.NoModifier:
            self._step(-1 if event.key() == Qt.Key_Left else 1)
            return
        super().keyPressEvent(event)
