"""
Strap Serial Check — pre-scan a sequential run of notes.

Uncirculated straps are often a sequential serial run. Rather than feed all 100
notes through the scanner and process them to find out whether any are fancy,
type the strap's FIRST serial and how many notes it holds; this walks the
sequence, classifies each serial against your currently-enabled patterns, and
lists which positions would be fancy — so you can decide whether the strap is
even worth scanning.

Digit-only patterns apply (year notes, radars, ladders, binaries, …). Patterns
that need the physical note (gas pump, seal shift) can't be predicted from the
number alone and are simply not part of this estimate.
"""

import re

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSpinBox, QPushButton,
    QTreeWidget, QTreeWidgetItem, QHeaderView,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush

# prefix letters (0-2), exactly 8 digits, optional suffix letter/star
_SERIAL_RE = re.compile(r"^\s*([A-Za-z]{0,2})\s*(\d{8})\s*([A-Za-z*]?)\s*$")


def parse_serial(text):
    """(prefix, number:int, suffix) or None."""
    m = _SERIAL_RE.match(text or "")
    if not m:
        return None
    return m.group(1).upper(), int(m.group(2)), m.group(3).upper()


def strap_serials(prefix, start_num, suffix, count):
    """Yield (position, full_serial) for a sequential run, wrapping the 8-digit
    counter at 100,000,000 (rare at a strap boundary)."""
    for i in range(count):
        num = (start_num + i) % 100_000_000
        yield i + 1, f"{prefix}{num:08d}{suffix}"


class StrapCheckDialog(QDialog):
    def __init__(self, engine, parent=None, start_serial=""):
        super().__init__(parent)
        self.engine = engine
        self.setWindowTitle("Strap Serial Check")
        self.resize(560, 560)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Type the strap's first serial and how many notes it holds. This "
            "lists which serials in that sequential run would be fancy — before "
            "you scan it."))

        row = QHBoxLayout()
        row.addWidget(QLabel("First serial:"))
        self.serial_edit = QLineEdit(start_serial)
        self.serial_edit.setPlaceholderText("e.g. H43699401C")
        self.serial_edit.returnPressed.connect(self._check)
        row.addWidget(self.serial_edit, 1)
        row.addWidget(QLabel("Count:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 10000)
        self.count_spin.setValue(100)
        row.addWidget(self.count_spin)
        self.check_btn = QPushButton("Check")
        self.check_btn.clicked.connect(self._check)
        row.addWidget(self.check_btn)
        layout.addLayout(row)

        self.summary = QLabel("")
        self.summary.setTextFormat(Qt.RichText)
        layout.addWidget(self.summary)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["#", "Serial", "Patterns"])
        self.tree.setRootIsDecorated(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.Stretch)
        layout.addWidget(self.tree, 1)

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        if start_serial:
            self._check()

    def _display_names(self, names):
        out = []
        for n in names:
            info = self.engine.get_pattern_info(n) if hasattr(self.engine, "get_pattern_info") else None
            out.append((info or {}).get("display_name") or n)
        return out

    def _check(self):
        parsed = parse_serial(self.serial_edit.text())
        if not parsed:
            self.summary.setText(
                "<b style='color:#c62828'>Enter a valid serial</b> — a letter or "
                "two, 8 digits, optional suffix (e.g. H43699401C).")
            self.tree.clear()
            return
        prefix, start_num, suffix = parsed
        count = self.count_spin.value()

        self.tree.clear()
        fancy = 0
        green = QBrush(QColor("#2e7d32"))
        for pos, serial in strap_serials(prefix, start_num, suffix, count):
            names = self.engine.classify_simple(serial)
            if not names:
                continue
            fancy += 1
            item = QTreeWidgetItem([str(pos), serial,
                                    ", ".join(self._display_names(names))])
            item.setForeground(1, green)
            self.tree.addTopLevelItem(item)

        last = f"{prefix}{(start_num + count - 1) % 100_000_000:08d}{suffix}"
        pct = (fancy / count * 100) if count else 0
        self.summary.setText(
            f"<b>{fancy}</b> of <b>{count}</b> serials would be fancy "
            f"(<b>{pct:.0f}%</b>) — run {prefix}{start_num:08d}{suffix} → {last}."
            + ("  <i>None — probably not worth scanning.</i>" if not fancy else ""))
