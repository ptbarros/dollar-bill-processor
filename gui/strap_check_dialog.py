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
    QTreeWidget, QTreeWidgetItem, QHeaderView, QProgressDialog, QApplication, QMenu,
)
from PySide6.QtCore import Qt, QRegularExpression
from PySide6.QtGui import QColor, QBrush, QRegularExpressionValidator

# prefix letters (0-2), exactly 8 digits, optional suffix letter/star
_SERIAL_RE = re.compile(r"^\s*([A-Za-z]{0,2})\s*(\d{8})\s*([A-Za-z*]?)\s*$")


def parse_serial(text):
    """(prefix, number:int, suffix) or None."""
    m = _SERIAL_RE.match(text or "")
    if not m:
        return None
    return m.group(1).upper(), int(m.group(2)), m.group(3).upper()


MAX_PRINTED_SERIAL = 96_000_000  # modern circulating max ($20 and below, Series 1988
                                 # on); above it = a pre-1988 note or an uncut-sheet serial


def strap_serials(prefix, start_num, suffix, count):
    """Yield (position, full_serial) for a strap of `count` consecutive notes,
    STOPPING at 96,000,000 — the modern circulating maximum ($20 and below,
    Series 1988 onward). Serials above it are pre-1988 notes (99,999,999 was
    standard until the 1970s) or uncut-sheet over-runs, not part of a modern
    sequential strap.
    The note after 96,000,000 is in a DIFFERENT block with different letters --
    not 00000001 with the same letters -- so the run ends rather than wrapping;
    wrapping would name a serial that isn't the next physical note in the strap."""
    for i in range(count):
        num = start_num + i
        if num > MAX_PRINTED_SERIAL:
            return
        yield i + 1, f"{prefix}{num:08d}{suffix}"


class _StrapItem(QTreeWidgetItem):
    """Tree row that sorts the numeric columns (# and Tier) by value, not text,
    so "10" sorts after "9" and a blank Tier sorts last."""
    _NUMERIC_COLS = (0, 2)  # "#" and "Tier"

    def __lt__(self, other):
        col = self.treeWidget().sortColumn() if self.treeWidget() else 0
        if col in self._NUMERIC_COLS:
            def num(item):
                try:
                    return float(item.text(col))
                except (ValueError, TypeError):
                    return float("inf")  # blanks / non-numeric sort last
            return num(self) < num(other)
        # Text columns (Serial, Patterns): compare the cell text directly --
        # super().__lt__ isn't reliable once __lt__ is overridden in PySide.
        return self.text(col).lower() < other.text(col).lower()


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
        self.serial_edit.setPlaceholderText("e.g. H43699401C  or  43699401")
        # Hard-cap the DIGIT count at 8 (a strap's fancy-ness is digit-based; the
        # letters don't matter here). Optional prefix/suffix letters are still
        # tolerated so a full serial can be pasted, but you can't type a 9th digit.
        self.serial_edit.setValidator(QRegularExpressionValidator(
            QRegularExpression(r"^[A-Za-z]{0,2}\d{0,8}[A-Za-z*]?$")))
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
        self.tree.setHeaderLabels(["#", "Serial", "Tier", "Patterns"])
        self.tree.setRootIsDecorated(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(3, QHeaderView.Stretch)
        # Sortable columns (# and Tier sort numerically via _StrapItem).
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(0, Qt.AscendingOrder)
        # Double-click / Enter looks the serial up; right-click copies it.
        self.tree.setToolTip("Double-click a row to open it in Serial Lookup; "
                             "right-click to copy the serial.")
        self.tree.itemActivated.connect(self._open_lookup)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_row_menu)
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
        if start_num > MAX_PRINTED_SERIAL:
            self.summary.setText(
                "<b style='color:#c62828'>Above the modern maximum</b> — modern serials "
                "($20 and below, Series 1988 on) stop at 96,000,000. "
                f"{prefix}{start_num:08d}{suffix} is a pre-1988 note or an uncut-sheet "
                "serial; look it up individually instead.")
            return

        fancy = 0
        checked = 0
        last_serial = f"{prefix}{start_num:08d}{suffix}"
        green = QBrush(QColor("#2e7d32"))

        # Progress dialog so a big strap (up to 10,000) doesn't look frozen. The
        # check runs on the GUI thread, so setValue() also pumps the event loop.
        progress = QProgressDialog("Checking strap…", "Cancel", 0, count, self)
        progress.setWindowTitle("Strap Serial Check")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(300)  # don't flash for small/fast runs

        self.tree.setSortingEnabled(False)  # bulk insert unsorted, re-enable after
        canceled = False
        for pos, serial in strap_serials(prefix, start_num, suffix, count):
            if progress.wasCanceled():
                canceled = True
                break
            checked += 1
            last_serial = serial
            matches = self.engine.classify(serial)
            progress.setValue(checked)
            if not matches:
                continue
            fancy += 1
            names = [m.name for m in matches]
            # classify() sorts by (tier, name), so matches[0] is the best (lowest)
            # tier -- the single value shown in the Tier column.
            tier = matches[0].tier
            item = _StrapItem([str(pos), serial, str(tier),
                               ", ".join(self._display_names(names))])
            item.setForeground(1, green)
            self.tree.addTopLevelItem(item)
        progress.setValue(count)
        self.tree.setSortingEnabled(True)

        pct = (fancy / checked * 100) if checked else 0
        msg = (f"<b>{fancy}</b> of <b>{checked}</b> serials would be fancy "
               f"(<b>{pct:.0f}%</b>) — run {prefix}{start_num:08d}{suffix} → {last_serial}.")
        if canceled:
            msg += f"  <i>Canceled at {checked} of {count}.</i>"
        elif checked < count:
            msg += (f"  <i>Run reaches 96,000,000, the modern circulating maximum; "
                    f"{checked} of {count} are modern serials. Any beyond would be "
                    f"pre-1988 notes or uncut-sheet serials, not part of a modern strap.</i>")
        elif not fancy:
            msg += "  <i>None — probably not worth scanning.</i>"
        self.summary.setText(msg)

    @staticmethod
    def _serial_of(item):
        return item.text(1) if item else ""

    def _open_lookup(self, item, column=0):
        """Open Serial Lookup prefilled with the row's serial. Parented to this
        dialog so it shows above the (modal) Strap Check window."""
        serial = self._serial_of(item)
        if not serial:
            return
        from gui.serial_lookup_dialog import SerialLookupDialog
        dlg = getattr(self, "_lookup_dialog", None)
        if dlg is None:
            dlg = SerialLookupDialog(self.engine, self)
            self._lookup_dialog = dlg
        dlg.serial_edit.setText(serial)  # textChanged runs the live lookup
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _show_row_menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        act_copy = menu.addAction("Copy serial")
        act_look = menu.addAction("Look up serial…")
        chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
        if chosen == act_copy:
            QApplication.clipboard().setText(self._serial_of(item))
        elif chosen == act_look:
            self._open_lookup(item)
