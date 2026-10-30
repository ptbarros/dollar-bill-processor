"""Find Crop by Serial — and what the ledger knows about it.

Type a serial number once and see everything Dollar Detective has on it:

  * the bill's **ledger history** — whether it's been seen before, how many
    times, first/last seen, whether it was kept or flagged fancy, which patterns
    fired, its plates, and a per-scan log; and
  * its **crop file(s)** wherever they were filed, so you don't depend on OS
    file search (which often can't see the Straps folder).

Crops are named "<serial>_NN.jpg" (star notes: '*' -> 'star'), so the crop search
walks the Straps tree for files matching the serial and lets you open the
containing folder or copy the path. The ledger side is a read-only query — no
reprocessing, no image or OCR involved.
"""
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTreeWidget, QTreeWidgetItem, QApplication, QHeaderView, QGroupBox,
    QSplitter,
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


def _pretty_ts(ts) -> str:
    """ISO timestamp -> 'YYYY-MM-DD HH:MM' for display (best-effort)."""
    if not ts:
        return "-"
    s = str(ts).replace("T", " ")
    return s[:16] if len(s) >= 16 else s


def _fmt_conf(val) -> str:
    try:
        return f"{float(val):.2f}"
    except (TypeError, ValueError):
        return "-"


class FindCropDialog(QDialog):
    """Non-modal: serial in -> ledger history + crop file(s) out."""

    def __init__(self, parent=None, initial_serial: str = "", ledger_provider=None):
        super().__init__(parent)
        # ledger_provider is a zero-arg callable returning a Ledger (or None).
        # Called fresh on each search so a ledger reopened after a Restore is
        # picked up without recreating this (cached) dialog.
        self._ledger_provider = ledger_provider
        self.setWindowTitle("Find Crop by Serial")
        self.setMinimumSize(680, 560)
        self.setModal(False)

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Type a serial number to see what Dollar Detective knows about it — "
            "its ledger history (seen before? kept? which patterns?) and its crop "
            "file(s). The crop search covers your filed straps, so it works even "
            "when Windows search can't see them.")
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

        split = QSplitter(Qt.Vertical)

        # -- Ledger history ------------------------------------------------
        hist_box = QGroupBox("Ledger history")
        hist_lay = QVBoxLayout(hist_box)
        self.history_summary = QLabel(" ")
        self.history_summary.setWordWrap(True)
        self.history_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.history_summary.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        hist_lay.addWidget(self.history_summary)
        self.obs_tree = QTreeWidget()
        self.obs_tree.setHeaderLabels(
            ["Scanned", "Batch", "Series", "Stack", "Read as", "Conf", "Front file"])
        self.obs_tree.setRootIsDecorated(False)
        self.obs_tree.setAlternatingRowColors(True)
        self.obs_tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.obs_tree.header().setStretchLastSection(True)
        hist_lay.addWidget(self.obs_tree, 1)
        split.addWidget(hist_box)

        # -- Crop files ----------------------------------------------------
        crop_box = QGroupBox("Crop files")
        crop_lay = QVBoxLayout(crop_box)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Serial", "Strap", "File"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.Stretch)
        self.tree.itemDoubleClicked.connect(lambda *_: self._open_folder())
        self.tree.itemSelectionChanged.connect(self._update_buttons)
        crop_lay.addWidget(self.tree, 1)
        split.addWidget(crop_box)

        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)

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

    # -- search --------------------------------------------------------------
    def _search(self):
        serial = self.serial_edit.text().strip()
        self.tree.clear()
        if not serial:
            self.status.setText(" ")
            self._render_history(None, "")
            self._update_buttons()
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._search_ledger(serial)
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

    def _search_ledger(self, serial: str):
        """Query the ledger for `serial` and render its history."""
        led = None
        if callable(self._ledger_provider):
            try:
                led = self._ledger_provider()
            except Exception:
                led = None
        if led is None:
            self._render_history(None, serial)
            return
        try:
            bills = led.lookup_serial(serial)
        except Exception as e:  # pragma: no cover - defensive
            self._render_history(None, serial, error=str(e))
            return
        self._render_history(bills, serial)

    def _render_history(self, bills, serial: str, error: str = ""):
        """Populate the summary label + observation tree from lookup_serial()."""
        self.obs_tree.clear()
        if error:
            self.history_summary.setText(
                f"<i>Couldn't read the ledger: {error}</i>")
            return
        if bills is None:
            self.history_summary.setText(
                "<i>The bill ledger isn't available, so no history can be shown.</i>")
            return
        if not bills:
            self.history_summary.setText(
                f"<b>Not in the ledger.</b> “{serial}” hasn't been processed on "
                "this machine yet (or was processed before the ledger existed).")
            return

        multi = len(bills) > 1
        blocks = []
        for b in bills:
            denom = b.get("denomination") or "$1"
            year = b.get("series_year") or ""
            head = denom + (f" · series {year}" if year else "")
            times = b.get("times_seen") or 0
            flags = []
            if b.get("kept"):
                kept_at = _pretty_ts(b.get("kept_at"))
                flags.append(f"<b>kept</b>{f' ({kept_at})' if kept_at != '-' else ''}")
            elif b.get("checked"):
                flags.append("checked")
            if b.get("is_fancy"):
                flags.append("fancy")
            if b.get("potential_mule"):
                flags.append("possible mule")

            lines = [f"<b>{head}</b> — seen {times}×, "
                     f"first {_pretty_ts(b.get('first_seen'))}, "
                     f"last {_pretty_ts(b.get('last_seen'))}"]
            if flags:
                lines.append("Status: " + ", ".join(flags))
            pats = b.get("patterns") or []
            lines.append("Patterns: " + (", ".join(pats) if pats else "<i>none</i>"))
            plate_bits = []
            if b.get("front_plate"):
                plate_bits.append(f"front {b['front_plate']}")
            if b.get("back_plate"):
                plate_bits.append(f"back {b['back_plate']}")
            if plate_bits:
                lines.append("Plates: " + ", ".join(plate_bits))
            if b.get("best_confidence") is not None:
                lines.append(f"Best OCR confidence: {_fmt_conf(b.get('best_confidence'))}")
            blocks.append("<br>".join(lines))

            for o in b.get("observations") or []:
                batch = o.get("session_label") or o.get("session_source") or "-"
                stack = o.get("stack_position")
                read_as = o.get("serial_read") or "-"
                front = o.get("front_file") or "-"
                item = QTreeWidgetItem([
                    _pretty_ts(o.get("seen_at")),
                    str(batch),
                    year or "-",
                    "" if stack is None else str(stack),
                    read_as,
                    _fmt_conf(o.get("confidence")),
                    front,
                ])
                if front and front != "-":
                    item.setToolTip(6, front)
                self.obs_tree.addTopLevelItem(item)

        header = (f"<b>Seen before</b> — {len(bills)} ledger records for this serial.<br>"
                  if multi else "")
        self.history_summary.setText(header + "<hr>".join(blocks))

    # -- crop actions --------------------------------------------------------
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
