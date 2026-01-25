"""
Preview Panel - Bill image preview and details.
"""

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QPushButton, QLineEdit, QFrame, QGroupBox, QGridLayout,
    QSizePolicy
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QPixmap, QImage

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from pattern_engine_v2 import PatternEngine


class ImageLabel(QLabel):
    """Label that displays an image with zoom capability."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 150)
        self.original_pixmap = None
        self.zoom_factor = 1.0

    def set_image(self, path: str):
        """Load and display an image."""
        if not path or not Path(path).exists():
            self.clear()
            self.setText("No image")
            return

        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.clear()
            self.setText("Failed to load image")
            return

        self.original_pixmap = pixmap
        self._update_display()

    def set_zoom(self, factor: float):
        """Set zoom factor (1.0 = fit to window)."""
        self.zoom_factor = factor
        self._update_display()

    def _update_display(self):
        """Update the displayed image."""
        if self.original_pixmap is None:
            return

        if self.zoom_factor == 1.0:
            # Fit to container
            scaled = self.original_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        else:
            # Apply zoom
            new_size = self.original_pixmap.size() * self.zoom_factor
            scaled = self.original_pixmap.scaled(
                new_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

        self.setPixmap(scaled)

    def resizeEvent(self, event):
        """Handle resize to update image scaling."""
        super().resizeEvent(event)
        if self.original_pixmap:
            self._update_display()


class PreviewPanel(QWidget):
    """Panel showing bill preview and correction interface."""

    # Signals
    correction_submitted = Signal(str, str, str)  # filename, original, corrected

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_result: Optional[dict] = None
        self.pattern_engine = PatternEngine()
        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Image preview area
        preview_group = QGroupBox("Bill Preview")
        preview_layout = QVBoxLayout(preview_group)

        # Main bill image
        self.bill_image = ImageLabel()
        self.bill_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.bill_image.setMinimumHeight(200)
        preview_layout.addWidget(self.bill_image, 1)

        # Serial region image
        serial_frame = QFrame()
        serial_layout = QHBoxLayout(serial_frame)
        serial_layout.setContentsMargins(0, 0, 0, 0)

        serial_label = QLabel("Serial Region:")
        serial_layout.addWidget(serial_label)

        self.serial_image = ImageLabel()
        self.serial_image.setMinimumSize(300, 60)
        self.serial_image.setMaximumHeight(80)
        serial_layout.addWidget(self.serial_image, 1)

        preview_layout.addWidget(serial_frame)

        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom:"))

        self.zoom_fit_btn = QPushButton("Fit")
        self.zoom_fit_btn.clicked.connect(lambda: self.bill_image.set_zoom(1.0))
        zoom_layout.addWidget(self.zoom_fit_btn)

        self.zoom_100_btn = QPushButton("100%")
        self.zoom_100_btn.clicked.connect(lambda: self.bill_image.set_zoom(1.5))
        zoom_layout.addWidget(self.zoom_100_btn)

        self.zoom_200_btn = QPushButton("200%")
        self.zoom_200_btn.clicked.connect(lambda: self.bill_image.set_zoom(2.0))
        zoom_layout.addWidget(self.zoom_200_btn)

        zoom_layout.addStretch()
        preview_layout.addLayout(zoom_layout)

        layout.addWidget(preview_group, 1)

        # Details section
        details_group = QGroupBox("Bill Details")
        details_layout = QGridLayout(details_group)

        # Serial number
        details_layout.addWidget(QLabel("Serial:"), 0, 0)
        self.serial_label = QLabel("-")
        self.serial_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.serial_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        details_layout.addWidget(self.serial_label, 0, 1)

        # Patterns
        details_layout.addWidget(QLabel("Patterns:"), 1, 0)
        self.patterns_label = QLabel("-")
        self.patterns_label.setWordWrap(True)
        details_layout.addWidget(self.patterns_label, 1, 1)

        # Rarity/Odds
        details_layout.addWidget(QLabel("Rarity:"), 2, 0)
        self.odds_label = QLabel("-")
        self.odds_label.setWordWrap(True)
        self.odds_label.setStyleSheet("color: #1976D2; font-weight: bold;")
        details_layout.addWidget(self.odds_label, 2, 1)

        # Confidence
        details_layout.addWidget(QLabel("Confidence:"), 3, 0)
        self.confidence_label = QLabel("-")
        details_layout.addWidget(self.confidence_label, 3, 1)

        # Status
        details_layout.addWidget(QLabel("Status:"), 4, 0)
        self.status_label = QLabel("-")
        details_layout.addWidget(self.status_label, 4, 1)

        # File info
        details_layout.addWidget(QLabel("File:"), 5, 0)
        self.file_label = QLabel("-")
        self.file_label.setWordWrap(True)
        details_layout.addWidget(self.file_label, 5, 1)

        layout.addWidget(details_group)

        # Correction section
        correction_group = QGroupBox("Serial Correction")
        correction_layout = QVBoxLayout(correction_group)

        # Correction input
        input_layout = QHBoxLayout()

        self.correction_edit = QLineEdit()
        self.correction_edit.setPlaceholderText("Enter corrected serial (e.g., C12345678A)")
        self.correction_edit.setMaxLength(10)
        self.correction_edit.returnPressed.connect(self._submit_correction)
        input_layout.addWidget(self.correction_edit, 1)

        self.submit_btn = QPushButton("Apply")
        self.submit_btn.clicked.connect(self._submit_correction)
        input_layout.addWidget(self.submit_btn)

        correction_layout.addLayout(input_layout)

        # Quick fix buttons
        quick_layout = QHBoxLayout()
        quick_layout.addWidget(QLabel("Quick fixes:"))

        # Common character confusions
        self.quick_fixes = [
            ("G->C", "G", "C"),
            ("C->G", "C", "G"),
            ("O->Q", "O", "Q"),
            ("Q->O", "Q", "O"),
            ("0->O", "0", "O"),
            ("O->0", "O", "0"),
            ("1->L", "1", "L"),
            ("L->1", "L", "1"),
            ("8->B", "8", "B"),
            ("B->8", "B", "8"),
        ]

        for label, from_char, to_char in self.quick_fixes:
            btn = QPushButton(label)
            btn.setMaximumWidth(60)
            btn.clicked.connect(lambda checked, f=from_char, t=to_char: self._apply_quick_fix(f, t))
            quick_layout.addWidget(btn)

        quick_layout.addStretch()
        correction_layout.addLayout(quick_layout)

        layout.addWidget(correction_group)

    def show_bill(self, result: dict):
        """Display a bill result."""
        self.current_result = result

        # Load images
        front_file = result.get('front_file', '')
        # Try to find the full path
        if front_file and not Path(front_file).exists():
            # Try relative to the input directory stored in result
            pass  # Path resolution would go here

        self.bill_image.set_image(front_file)

        # Serial region image if available
        serial_region = result.get('serial_region_path', '')
        self.serial_image.set_image(serial_region)

        # Update details
        serial = result.get('serial', '')
        if result.get('corrected'):
            serial += " (corrected)"
        self.serial_label.setText(serial or "-")

        patterns = result.get('fancy_types', '')
        self.patterns_label.setText(patterns or "None")

        # Look up odds for matched patterns
        odds_parts = []
        if patterns:
            pattern_names = [p.strip() for p in patterns.split(',')]
            for name in pattern_names:
                info = self.pattern_engine.get_pattern_info(name)
                if info and 'odds' in info:
                    odds_parts.append(f"{name}: {info['odds']}")
        if odds_parts:
            self.odds_label.setText('\n'.join(odds_parts))
        else:
            self.odds_label.setText("-")

        conf = result.get('confidence', 0)
        self.confidence_label.setText(f"{conf}" if conf else "-")

        # Status
        status_parts = []
        if result.get('is_fancy'):
            status_parts.append("Fancy")
        if result.get('needs_review'):
            status_parts.append("Needs Review")
        if result.get('error'):
            status_parts.append(f"Error: {result['error']}")
        self.status_label.setText(', '.join(status_parts) if status_parts else "OK")

        self.file_label.setText(front_file or "-")

        # Pre-fill correction with current serial
        self.correction_edit.setText(result.get('serial', ''))

    def start_correction(self):
        """Start correction mode - focus on the edit field."""
        self.correction_edit.setFocus()
        self.correction_edit.selectAll()

    def _submit_correction(self):
        """Submit the correction."""
        if not self.current_result:
            return

        corrected = self.correction_edit.text().strip().upper()
        if not corrected:
            return

        # Validate format
        import re
        if not re.match(r'^[A-L]\d{8}[A-Y*]$', corrected):
            self.status_label.setText("Invalid serial format!")
            self.status_label.setStyleSheet("color: red;")
            return

        filename = self.current_result.get('front_file', '')
        original = self.current_result.get('serial', '')

        self.correction_submitted.emit(filename, original, corrected)

        # Update display
        self.serial_label.setText(f"{corrected} (corrected)")
        self.status_label.setText("Correction saved")
        self.status_label.setStyleSheet("color: green;")

    def _apply_quick_fix(self, from_char: str, to_char: str):
        """Apply a quick character replacement."""
        current = self.correction_edit.text()
        # Replace first occurrence of from_char
        if from_char in current:
            fixed = current.replace(from_char, to_char, 1)
            self.correction_edit.setText(fixed)
        self.correction_edit.setFocus()
