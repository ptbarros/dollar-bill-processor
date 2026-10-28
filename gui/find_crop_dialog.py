"""Find Crop by Serial.

Type a serial number and locate its eBay crop file(s) wherever they were filed,
so you don't depend on OS file search (which often can't see the Straps folder).
Crops are named "<serial>_NN.jpg" (star notes: '*' -> 'star'), so this searches
the Straps tree for files matching the serial and lets you open the containing
folder or copy the path.
"""
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTreeWidget, QTreeWidgetItem, QApplication, QHeaderView,
)

from dollar_detective.resource_path import straps_dir, content_dir
from dollar_detective.process_production import _safe_serial_for_filename

_IMG_EXTS = {".jpg", ".jpeg", ".png"}
_PATH_ROLE = Qt.UserRole


def reveal_in_file_manager(path: Path) -> None:
    """Open the OS file manager with `path` selected (or its folder opened)."""
    p = str(path)
    try:
        if sys.platform == "win32":
            # explorer returns exit code 1 even on success -> don't check.
            subprocess.run(["explorer", f"/select,{p}"])
        elif sys.platform == "darwin":
            subprocess.run(["open", "-R", p])
        else:
            subprocess.run(["xdg-open", str(Path(path).parent)])
    except Exception:
        pass


def find_crops_for_serial(serial: str, roots=None):
    """Return a sorted list of crop file Paths whose name matches `serial`.

    Matches on the sanitized serial prefix, so a partial serial also works and
    star notes match whether the user types '*' or 'star'. Searches the Straps
    tree by default.
    """
    # Sanitize the same way crops are named ('*' -> 'star'), then upper-case so
    # matching is case-insensitive even for the inserted 'star'.
    safe = _safe_serial_for_filename((serial or "").strip().upper()).upper()
    if not safe or safe == "UNKNOWN":
        return []
    if roots is None:
        roots = []
        for getter in (straps_dir, content_dir):
            try:
                d = getter()
                if d and d not in roots:
                    roots.append(d)
            except Exception:
                pass
    seen, out = set(), []
    for root in roots:
        try:
            root = Path(root)
            if not root.exists():
                continue
            for p in root.rglob("*"):
                if (p.suffix.lower() in _IMG_EXTS and p.is_file()
                        and p.stem.upper().startswith(safe)):
                    rp = p.resolve()
                    if rp not in seen:
                        seen.add(rp)
                        out.append(p)
        except Exception:
            continue
    return sorted(out, key=lambda p: (p.parent.as_posix(), p.name))


class FindCropDialog(QDialog):
    """Non-modal: find a serial's crop file(s) and reveal/copy them."""

    def __init__(self, parent=None, initial_serial: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Find Crop by Serial")
        self.setMinimumSize(640, 420)
        self.setModal(False)

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Type a serial number to find its crop file(s). Searches your filed "
            "straps, so it works even when Windows search can't see them.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        row = QHBoxLayout()
        row.addWidget(QLabel("Serial:"))
        self.serial_edit = QLineEdit()
        self.serial_edit.setPlaceholderText("e.g. B01430046*  (partial serials work too)")
        self.serial_edit.returnPressed.connect(self._search)
        row.addWidget(self.serial_edit, 1)
        self.find_btn = QPushButton("Find")
        self.find_btn.setDefault(True)
        self.find_btn.clicked.connect(self._search)
        row.addWidget(self.find_btn)
        layout.addLayout(row)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Serial", "Strap", "File"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.Stretch)
        self.tree.itemDoubleClicked.connect(lambda *_: self._open_folder())
        self.tree.itemSelectionChanged.connect(self._update_buttons)
        layout.addWidget(self.tree, 1)

        self.status = QLabel(" ")
        layout.addWidget(self.status)

        btns = QHBoxLayout()
        self.open_btn = QPushButton("Open Folder")
        self.open_btn.setToolTip("Reveal the selected crop in your file manager")
        self.open_btn.clicked.connect(self._open_folder)
        self.copy_btn = QPushButton("Copy Path")
        self.copy_btn.clicked.connect(self._copy_path)
        btns.addWidget(self.open_btn)
        btns.addWidget(self.copy_btn)
        btns.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btns.addWidget(close_btn)
        layout.addLayout(btns)

        self._update_buttons()
        if initial_serial:
            self.serial_edit.setText(initial_serial)
            self._search()
        self.serial_edit.setFocus()

    def _search(self):
        serial = self.serial_edit.text().strip()
        self.tree.clear()
        if not serial:
            self.status.setText(" ")
            self._update_buttons()
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            matches = find_crops_for_serial(serial)
        finally:
            QApplication.restoreOverrideCursor()

        try:
            straps_root = straps_dir()
        except Exception:
            straps_root = None
        for p in matches:
            strap = "-"
            try:
                if straps_root and straps_root in p.parents:
                    strap = p.relative_to(straps_root).parts[0]
                else:
                    strap = p.parent.name
            except Exception:
                strap = p.parent.name
            item = QTreeWidgetItem([p.stem, strap, p.name])
            item.setData(0, _PATH_ROLE, str(p))
            item.setToolTip(2, str(p))
            self.tree.addTopLevelItem(item)

        n = len(matches)
        if n:
            self.tree.setCurrentItem(self.tree.topLevelItem(0))
            self.status.setText(f"{n} crop{'s' if n != 1 else ''} found for “{serial}”")
        else:
            self.status.setText(
                f"No crops found for “{serial}”. "
                "It may not have been cropped yet, or it's outside your Straps folder.")
        self._update_buttons()

    def _selected_path(self):
        item = self.tree.currentItem()
        if not item:
            return None
        p = item.data(0, _PATH_ROLE)
        return Path(p) if p else None

    def _update_buttons(self):
        has = self._selected_path() is not None
        self.open_btn.setEnabled(has)
        self.copy_btn.setEnabled(has)

    def _open_folder(self):
        p = self._selected_path()
        if p:
            reveal_in_file_manager(p)

    def _copy_path(self):
        p = self._selected_path()
        if p:
            QApplication.clipboard().setText(str(p))
            self.status.setText(f"Copied path: {p}")
