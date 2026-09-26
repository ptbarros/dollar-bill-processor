"""
Scan Mode Setup Wizard.

A short, friendly first-run walk-through for Scan mode. It explains — in plain
language — the folders and strap naming that Scan mode needs, and writes the
answers into the SAME settings the Folders tab uses (processing.data_folder,
processing.archive_directory, processing.batch_name_format, processing.batch_counter).

Offered automatically the first time a user enters Scan mode (once), and available
any time from Help → Setup Wizard.
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QSpinBox, QFileDialog
)
from PySide6.QtCore import Qt

sys.path.insert(0, str(Path(__file__).parent.parent))

# Strap-name formats, in the same order the Folders tab lists them.
_FMT_VALUES = ["number", "date", "number_date", "date_number"]
_FMT_LABELS = [
    "Number  (001)",
    "Date  (2026-09-25)",
    "Number + Date  (001 - 2026-09-25)",
    "Date + Number  (2026-09-25 - 001)",
]


def _hint(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color: gray; font-size: 11px;")
    return lbl


class _IntroPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Welcome to Scan mode")
        layout = QVBoxLayout(self)
        body = QLabel(
            "Scan mode lets Dollar Detective <b>watch your scanner's output folder</b> "
            "and file each strap for you.\n\n"
            "You feed a strap on the scanner (in as many passes as you like), then click "
            "<b>Stop &amp; File Batch</b> — everything that arrived is filed as the next "
            "strap and processed.\n\n"
            "This quick setup just points the app at the right folders and sets how your "
            "straps are named and numbered. You can change any of it later under "
            "<b>Settings → Folders</b>."
        )
        body.setWordWrap(True)
        body.setTextFormat(Qt.RichText)
        layout.addWidget(body)
        layout.addStretch()


class _FolderPage(QWizardPage):
    """A page with one explanation + a folder field and a Browse button."""

    def __init__(self, title, explanation, hint, initial, placeholder):
        super().__init__()
        self.setTitle(title)
        layout = QVBoxLayout(self)
        body = QLabel(explanation)
        body.setWordWrap(True)
        body.setTextFormat(Qt.RichText)
        layout.addWidget(body)

        row = QHBoxLayout()
        self.edit = QLineEdit(initial or "")
        self.edit.setPlaceholderText(placeholder)
        row.addWidget(self.edit)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        layout.addLayout(row)

        layout.addWidget(_hint(hint))
        layout.addStretch()

    def _browse(self):
        start = self.edit.text().strip() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", start)
        if folder:
            self.edit.setText(folder)

    def value(self) -> str:
        return self.edit.text().strip()


class _NamingPage(QWizardPage):
    def __init__(self, fmt_value, next_number):
        super().__init__()
        self.setTitle("Strap names & numbering")
        layout = QVBoxLayout(self)
        body = QLabel(
            "Each filed strap gets its own folder. Choose how those folders are named, "
            "and — if you number them — what number to start from.\n\n"
            "Set the start number to continue an existing sequence (for example, if your "
            "last strap was 823, start at 824)."
        )
        body.setWordWrap(True)
        layout.addWidget(body)

        form = QHBoxLayout()
        form.addWidget(QLabel("Strap Folder Names:"))
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(_FMT_LABELS)
        try:
            self.fmt_combo.setCurrentIndex(_FMT_VALUES.index(fmt_value))
        except (ValueError, TypeError):
            self.fmt_combo.setCurrentIndex(0)
        form.addWidget(self.fmt_combo)
        form.addStretch()
        layout.addLayout(form)

        numrow = QHBoxLayout()
        numrow.addWidget(QLabel("Strap Start Number:"))
        self.num_spin = QSpinBox()
        self.num_spin.setRange(1, 999999)
        self.num_spin.setValue(int(next_number or 1))
        numrow.addWidget(self.num_spin)
        numrow.addStretch()
        layout.addLayout(numrow)

        layout.addWidget(_hint(
            "The start number applies to the Number formats. Names can always be "
            "changed later in Settings → Folders."))
        layout.addStretch()

    def fmt_value(self) -> str:
        return _FMT_VALUES[self.fmt_combo.currentIndex()]

    def start_number(self) -> int:
        return self.num_spin.value()


class _SummaryPage(QWizardPage):
    def __init__(self, wizard):
        super().__init__()
        self._wiz = wizard
        self.setTitle("All set")
        layout = QVBoxLayout(self)
        self.body = QLabel()
        self.body.setWordWrap(True)
        self.body.setTextFormat(Qt.RichText)
        layout.addWidget(self.body)
        layout.addStretch()

    def initializePage(self):
        w = self._wiz
        scanner = w.scanner_page.value() or "(default: DollarDetective in your home folder)"
        straps = w.straps_page.value() or "(default: a 'Straps' subfolder of the Scanner Output Folder)"
        fmt_label = _FMT_LABELS[_FMT_VALUES.index(w.naming_page.fmt_value())]
        num = w.naming_page.start_number()
        self.body.setText(
            "Here's how Scan mode is set up:<br><br>"
            f"&nbsp;&nbsp;<b>Scanner Output Folder:</b><br>&nbsp;&nbsp;{scanner}<br><br>"
            f"&nbsp;&nbsp;<b>Straps Folder:</b><br>&nbsp;&nbsp;{straps}<br><br>"
            f"&nbsp;&nbsp;<b>Strap Folder Names:</b> {fmt_label}<br>"
            f"&nbsp;&nbsp;<b>Next strap number:</b> {num}<br><br>"
            "Click <b>Finish</b> to save. Then click <b>Start Scanning</b> and feed a "
            "strap. You can revisit any of this under Settings → Folders, or re-run this "
            "wizard from Help → Setup Wizard."
        )


class ScanSetupWizard(QWizard):
    """First-run setup for Scan mode. Reads/writes the same processing settings as
    the Folders tab. Call exec(); on accept it saves and returns QDialog.Accepted."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Scan Mode Setup")
        self.setWizardStyle(QWizard.ModernStyle)
        self.setOption(QWizard.NoBackButtonOnStartPage, True)

        # Resolve current values / sensible defaults to pre-fill.
        try:
            from resource_path import content_dir, straps_dir
            default_scanner = str(content_dir())
            default_straps = str(straps_dir())
        except Exception:
            default_scanner = str(Path.home() / "DollarDetective")
            default_straps = ""

        p = settings.processing
        next_number = int(getattr(p, "batch_counter", 0) or 0) + 1

        self.addPage(_IntroPage())

        self.scanner_page = _FolderPage(
            "Scanner Output Folder",
            "This is the folder <b>where your scanner saves its scanned images</b> — the "
            "one Scan mode watches while you feed a strap.<br><br>"
            "Set your scanner software to save here, then pick that same folder below.",
            f"Leave blank to use the default: {default_scanner}",
            initial=getattr(p, "data_folder", "") or "",
            placeholder=default_scanner,
        )
        self.addPage(self.scanner_page)

        self.straps_page = _FolderPage(
            "Straps Folder",
            "This is <b>where finished straps are filed</b> — each strap becomes its own "
            "numbered folder here, holding its bills, crops, and results.<br><br>"
            "Most people leave this blank and let the app keep straps in a 'Straps' "
            "subfolder of the Scanner Output Folder.",
            f"Leave blank to use the default: {default_straps}",
            initial=getattr(p, "archive_directory", "") or "",
            placeholder=default_straps or "Default: a 'Straps' subfolder",
        )
        self.addPage(self.straps_page)

        self.naming_page = _NamingPage(
            getattr(p, "batch_name_format", "number") or "number", next_number)
        self.addPage(self.naming_page)

        self.addPage(_SummaryPage(self))

    def accept(self):
        """Persist the wizard's answers into processing settings."""
        p = self.settings.processing
        p.data_folder = self.scanner_page.value()
        p.archive_directory = self.straps_page.value()
        p.batch_name_format = self.naming_page.fmt_value()
        # Stored counter is (next number - 1); next_batch_number() pre-increments it.
        p.batch_counter = max(0, self.naming_page.start_number() - 1)
        try:
            self.settings.ui.scan_wizard_seen = True
            self.settings.save()
        except Exception:
            pass
        super().accept()
