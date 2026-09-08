"""
Backup / Restore dialogs (File -> Back Up Data / Restore from Backup).

Thin UI over backup_manager: pick categories, and for restore choose how the
bill ledger comes back (merge vs replace). Logic lives in backup_manager.py.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QGroupBox,
    QRadioButton, QDialogButtonBox, QWidget,
)
from PySide6.QtCore import Qt

from backup_manager import CATEGORIES, CATEGORY_LABELS


class BackupDialog(QDialog):
    """Choose what to include in a backup."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Back Up Data")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)

        intro = QLabel(
            "Choose what to save into one portable backup file. Your scan images "
            "and archived originals are not included — keep those with your "
            "normal file backups."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        box = QGroupBox("Include")
        box_l = QVBoxLayout(box)
        self._checks = {}
        for cat in CATEGORIES:
            cb = QCheckBox(CATEGORY_LABELS[cat])
            cb.setChecked(True)
            self._checks[cat] = cb
            box_l.addWidget(cb)
        layout.addWidget(box)

        self._keys = QCheckBox("Include saved AI API keys")
        self._keys.setChecked(True)
        self._keys.setToolTip("Your Anthropic/OpenAI keys live in settings. "
                              "Uncheck to leave them out of the backup file.")
        layout.addWidget(self._keys)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("Choose File…")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_categories(self):
        return [c for c, cb in self._checks.items() if cb.isChecked()]

    def include_keys(self) -> bool:
        return self._keys.isChecked()


class RestoreDialog(QDialog):
    """Choose which categories to restore from a backup, and the ledger mode."""

    def __init__(self, manifest: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Restore from Backup")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)

        present = manifest.get("categories", {})
        info = QLabel(
            f"<b>Backup contents</b><br>"
            f"Created: {manifest.get('created', '?')}<br>"
            f"From: {manifest.get('source_os', '?')} · app {manifest.get('app_version', '?')}"
        )
        info.setTextFormat(Qt.RichText)
        layout.addWidget(info)

        box = QGroupBox("Restore")
        box_l = QVBoxLayout(box)
        self._checks = {}
        for cat in CATEGORIES:
            if cat not in present:
                continue
            n = present[cat].get("count", 0)
            cb = QCheckBox(f"{CATEGORY_LABELS[cat]}  ({n} item{'s' if n != 1 else ''})")
            cb.setChecked(True)
            self._checks[cat] = cb
            box_l.addWidget(cb)
        layout.addWidget(box)

        self._merge = None
        self._replace = None
        if "ledger" in present:
            lbox = QGroupBox("Bill ledger")
            lb = QVBoxLayout(lbox)
            self._merge = QRadioButton("Merge — combine the backup's history with what's here now")
            self._replace = QRadioButton("Replace — overwrite the current ledger with the backup")
            self._merge.setChecked(True)
            lb.addWidget(self._merge)
            lb.addWidget(self._replace)
            layout.addWidget(lbox)

        note = QLabel(
            "Settings, crop profiles and corrections are replaced by the backup. "
            "A safety copy of your current data is saved first, and file paths "
            "that don't exist on this computer are cleared. "
            "<b>Restart the app after restoring.</b>"
        )
        note.setWordWrap(True)
        note.setTextFormat(Qt.RichText)
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Restore")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_categories(self):
        return [c for c, cb in self._checks.items() if cb.isChecked()]

    def ledger_mode(self) -> str:
        if self._replace is not None and self._replace.isChecked():
            return "replace"
        return "merge"
