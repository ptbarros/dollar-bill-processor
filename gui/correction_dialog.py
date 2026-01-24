"""
Correction Dialog - Modal dialog for correcting serial numbers.
"""

import re
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QDialogButtonBox, QGroupBox, QGridLayout,
    QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap


class CorrectionDialog(QDialog):
    """
    Dialog for correcting OCR misreads.

    Usage:
        dialog = CorrectionDialog(
            serial="G12345678A",
            image_path="/path/to/serial_region.jpg",
            parent=self
        )
        if dialog.exec() == QDialog.Accepted:
            corrected = dialog.get_corrected_serial()
    """

    def __init__(
        self,
        serial: str = "",
        image_path: str = "",
        filename: str = "",
        parent=None
    ):
        super().__init__(parent)
        self.original_serial = serial
        self.corrected_serial = serial
        self.image_path = image_path
        self.filename = filename

        self.setWindowTitle("Correct Serial Number")
        self.setMinimumWidth(500)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)

        # File info
        if self.filename:
            file_label = QLabel(f"File: {self.filename}")
            file_label.setStyleSheet("color: gray;")
            layout.addWidget(file_label)

        # Image preview
        if self.image_path:
            image_group = QGroupBox("Serial Region")
            image_layout = QVBoxLayout(image_group)

            self.image_label = QLabel()
            self.image_label.setAlignment(Qt.AlignCenter)
            pixmap = QPixmap(self.image_path)
            if not pixmap.isNull():
                # Scale to reasonable size
                scaled = pixmap.scaledToHeight(100, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled)
            else:
                self.image_label.setText("(No image available)")

            image_layout.addWidget(self.image_label)
            layout.addWidget(image_group)

        # Original serial
        original_layout = QHBoxLayout()
        original_layout.addWidget(QLabel("Original read:"))
        self.original_label = QLabel(self.original_serial or "(none)")
        self.original_label.setStyleSheet("font-weight: bold; font-family: monospace;")
        original_layout.addWidget(self.original_label)
        original_layout.addStretch()
        layout.addLayout(original_layout)

        # Correction input
        correction_group = QGroupBox("Corrected Serial")
        correction_layout = QVBoxLayout(correction_group)

        self.correction_edit = QLineEdit(self.original_serial)
        self.correction_edit.setMaxLength(10)
        self.correction_edit.setStyleSheet("font-size: 18px; font-family: monospace;")
        self.correction_edit.textChanged.connect(self._validate_input)
        correction_layout.addWidget(self.correction_edit)

        # Validation feedback
        self.validation_label = QLabel("")
        self.validation_label.setStyleSheet("color: gray;")
        correction_layout.addWidget(self.validation_label)

        layout.addWidget(correction_group)

        # Quick fix buttons
        quick_group = QGroupBox("Quick Fixes")
        quick_layout = QGridLayout(quick_group)

        fixes = [
            ("G <-> C", self._swap_gc),
            ("0 <-> O", self._swap_0o),
            ("1 <-> L", self._swap_1l),
            ("8 <-> B", self._swap_8b),
            ("5 <-> S", self._swap_5s),
            ("6 <-> G", self._swap_6g),
        ]

        for i, (label, func) in enumerate(fixes):
            btn = QPushButton(label)
            btn.clicked.connect(func)
            quick_layout.addWidget(btn, i // 3, i % 3)

        layout.addWidget(quick_group)

        # Character reference
        ref_label = QLabel(
            "Format: [A-L] + 8 digits + [A-Y or *]\n"
            "Example: B12345678A or B12345678*"
        )
        ref_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(ref_label)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        self.ok_button = button_box.button(QDialogButtonBox.Ok)
        layout.addWidget(button_box)

        # Initial validation
        self._validate_input()

    def _validate_input(self):
        """Validate the current input."""
        text = self.correction_edit.text().upper()
        self.correction_edit.blockSignals(True)
        self.correction_edit.setText(text)
        self.correction_edit.blockSignals(False)

        if not text:
            self.validation_label.setText("Enter a serial number")
            self.validation_label.setStyleSheet("color: orange;")
            self.ok_button.setEnabled(False)
            return

        if len(text) != 10:
            self.validation_label.setText(f"Length: {len(text)}/10")
            self.validation_label.setStyleSheet("color: orange;")
            self.ok_button.setEnabled(False)
            return

        # Check format
        if not re.match(r'^[A-L]\d{8}[A-Y*]$', text):
            # Give specific feedback
            if text[0] not in 'ABCDEFGHIJKL':
                self.validation_label.setText(f"First letter must be A-L (got: {text[0]})")
            elif not text[1:9].isdigit():
                self.validation_label.setText("Middle 8 characters must be digits")
            elif text[-1] not in 'ABCDEFGHIJKLMNPQRSTUVWXY*':
                self.validation_label.setText(f"Last character must be A-Y or * (got: {text[-1]})")
            self.validation_label.setStyleSheet("color: red;")
            self.ok_button.setEnabled(False)
            return

        # Valid!
        self.validation_label.setText("Valid serial format")
        self.validation_label.setStyleSheet("color: green;")
        self.ok_button.setEnabled(True)
        self.corrected_serial = text

    def _swap_gc(self):
        """Swap G and C."""
        self._swap_chars('G', 'C')

    def _swap_0o(self):
        """Swap 0 and O."""
        self._swap_chars('0', 'O')

    def _swap_1l(self):
        """Swap 1 and L."""
        self._swap_chars('1', 'L')

    def _swap_8b(self):
        """Swap 8 and B."""
        self._swap_chars('8', 'B')

    def _swap_5s(self):
        """Swap 5 and S."""
        self._swap_chars('5', 'S')

    def _swap_6g(self):
        """Swap 6 and G."""
        self._swap_chars('6', 'G')

    def _swap_chars(self, char1: str, char2: str):
        """Swap two characters in the text."""
        text = self.correction_edit.text()
        # Use a placeholder to do the swap
        temp = chr(1)  # Unlikely character
        text = text.replace(char1, temp)
        text = text.replace(char2, char1)
        text = text.replace(temp, char2)
        self.correction_edit.setText(text)

    def _on_accept(self):
        """Handle accept button."""
        self.corrected_serial = self.correction_edit.text().upper()
        self.accept()

    def get_corrected_serial(self) -> str:
        """Get the corrected serial number."""
        return self.corrected_serial


# Test dialog standalone
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    dialog = CorrectionDialog(
        serial="G12345678A",
        filename="test_bill.jpg"
    )

    if dialog.exec() == QDialog.Accepted:
        print(f"Corrected: {dialog.get_corrected_serial()}")
    else:
        print("Cancelled")
