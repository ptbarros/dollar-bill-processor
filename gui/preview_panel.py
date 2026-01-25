"""
Preview Panel - Bill image preview and details with zoom/pan support.
"""

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QPushButton, QLineEdit, QFrame, QGroupBox, QGridLayout,
    QSizePolicy, QTabWidget, QSlider, QApplication
)
from PySide6.QtCore import Qt, Signal, Slot, QPoint, QSize
from PySide6.QtGui import QPixmap, QImage, QMouseEvent, QWheelEvent, QCursor

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from pattern_engine_v2 import PatternEngine


class PannableImageLabel(QLabel):
    """Label that displays an image with pan support via mouse drag."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self._drag_start = None
        self._scroll_area = None

    def set_scroll_area(self, scroll_area: QScrollArea):
        """Set the parent scroll area for panning."""
        self._scroll_area = scroll_area

    def mousePressEvent(self, event: QMouseEvent):
        """Start panning on mouse press."""
        if event.button() == Qt.LeftButton:
            self._drag_start = event.position().toPoint()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Pan the image while dragging."""
        if self._drag_start and self._scroll_area:
            delta = event.position().toPoint() - self._drag_start
            h_bar = self._scroll_area.horizontalScrollBar()
            v_bar = self._scroll_area.verticalScrollBar()
            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())
            self._drag_start = event.position().toPoint()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """End panning on mouse release."""
        if event.button() == Qt.LeftButton:
            self._drag_start = None
            self.setCursor(QCursor(Qt.OpenHandCursor))
        super().mouseReleaseEvent(event)


class ScrollableImageViewer(QWidget):
    """Image viewer with zoom and pan capabilities."""

    zoom_changed = Signal(int)  # Emits zoom percentage

    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = None
        self.zoom_factor = 1.0
        self._setup_ui()

    def _setup_ui(self):
        """Setup the scrollable image viewer."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Scroll area for panning
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setStyleSheet("QScrollArea { background-color: #2d2d2d; }")

        # Image label
        self.image_label = PannableImageLabel()
        self.image_label.set_scroll_area(self.scroll_area)
        self.image_label.setCursor(QCursor(Qt.OpenHandCursor))
        self.scroll_area.setWidget(self.image_label)

        layout.addWidget(self.scroll_area, 1)

        # Zoom controls
        zoom_layout = QHBoxLayout()

        self.zoom_fit_btn = QPushButton("Fit")
        self.zoom_fit_btn.setMaximumWidth(50)
        self.zoom_fit_btn.clicked.connect(self._zoom_fit)
        zoom_layout.addWidget(self.zoom_fit_btn)

        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setMaximumWidth(30)
        self.zoom_out_btn.clicked.connect(self._zoom_out)
        zoom_layout.addWidget(self.zoom_out_btn)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.zoom_slider.setTickInterval(50)
        self.zoom_slider.valueChanged.connect(self._on_slider_changed)
        zoom_layout.addWidget(self.zoom_slider, 1)

        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setMaximumWidth(30)
        self.zoom_in_btn.clicked.connect(self._zoom_in)
        zoom_layout.addWidget(self.zoom_in_btn)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(45)
        zoom_layout.addWidget(self.zoom_label)

        layout.addLayout(zoom_layout)

    def set_image(self, path: str):
        """Load and display an image."""
        if not path or not Path(path).exists():
            self.original_pixmap = None
            self.image_label.clear()
            self.image_label.setText("No image")
            return

        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.original_pixmap = None
            self.image_label.clear()
            self.image_label.setText("Failed to load image")
            return

        self.original_pixmap = pixmap
        self._zoom_fit()

    def _update_display(self):
        """Update the displayed image based on zoom level."""
        if self.original_pixmap is None:
            return

        new_size = self.original_pixmap.size() * self.zoom_factor
        scaled = self.original_pixmap.scaled(
            new_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)
        self.image_label.resize(scaled.size())

        # Update zoom label
        percent = int(self.zoom_factor * 100)
        self.zoom_label.setText(f"{percent}%")

        # Update slider without triggering signal
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(percent)
        self.zoom_slider.blockSignals(False)

        self.zoom_changed.emit(percent)

    def _zoom_fit(self):
        """Fit image to viewport."""
        if self.original_pixmap is None:
            return

        viewport_size = self.scroll_area.viewport().size()
        img_size = self.original_pixmap.size()

        # Calculate scale to fit
        scale_w = viewport_size.width() / img_size.width()
        scale_h = viewport_size.height() / img_size.height()
        self.zoom_factor = min(scale_w, scale_h) * 0.95  # 95% to leave margin

        self._update_display()

    def _zoom_in(self):
        """Zoom in by 25%."""
        self.zoom_factor = min(4.0, self.zoom_factor * 1.25)
        self._update_display()

    def _zoom_out(self):
        """Zoom out by 25%."""
        self.zoom_factor = max(0.1, self.zoom_factor / 1.25)
        self._update_display()

    def _on_slider_changed(self, value):
        """Handle zoom slider change."""
        self.zoom_factor = value / 100.0
        self._update_display()

    def set_zoom(self, factor: float):
        """Set zoom factor directly."""
        self.zoom_factor = max(0.1, min(4.0, factor))
        self._update_display()

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        if event.modifiers() == Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self._zoom_in()
            else:
                self._zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)

    def resizeEvent(self, event):
        """Handle resize - refit if at fit zoom."""
        super().resizeEvent(event)
        # Only auto-fit if we're close to fit zoom
        if self.original_pixmap and self.zoom_factor < 1.0:
            self._zoom_fit()


class ImageLabel(QLabel):
    """Simple label for small images like serial region."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 60)
        self.original_pixmap = None

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

    def _update_display(self):
        """Update the displayed image scaled to fit."""
        if self.original_pixmap is None:
            return

        scaled = self.original_pixmap.scaled(
            self.size(),
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

        # Tabbed image preview area
        preview_group = QGroupBox("Bill Preview")
        preview_layout = QVBoxLayout(preview_group)

        # Tab widget for Front/Back
        self.image_tabs = QTabWidget()

        # Front tab
        self.front_viewer = ScrollableImageViewer()
        self.image_tabs.addTab(self.front_viewer, "Front")

        # Back tab
        self.back_viewer = ScrollableImageViewer()
        self.image_tabs.addTab(self.back_viewer, "Back")

        preview_layout.addWidget(self.image_tabs, 1)

        # Serial region image (always visible)
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

        # Zoom tip
        zoom_tip = QLabel("Tip: Ctrl+Scroll to zoom, drag to pan")
        zoom_tip.setStyleSheet("color: gray; font-size: 10px;")
        preview_layout.addWidget(zoom_tip)

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
            ("G→C", "G", "C"),
            ("C→G", "C", "G"),
            ("O→Q", "O", "Q"),
            ("Q→O", "Q", "O"),
            ("0→O", "0", "O"),
            ("O→0", "O", "0"),
            ("1→L", "1", "L"),
            ("L→1", "L", "1"),
            ("8→B", "8", "B"),
            ("B→8", "B", "8"),
        ]

        for label, from_char, to_char in self.quick_fixes:
            btn = QPushButton(label)
            btn.setMaximumWidth(50)
            btn.clicked.connect(lambda checked, f=from_char, t=to_char: self._apply_quick_fix(f, t))
            quick_layout.addWidget(btn)

        quick_layout.addStretch()
        correction_layout.addLayout(quick_layout)

        layout.addWidget(correction_group)

    def show_bill(self, result: dict):
        """Display a bill result."""
        self.current_result = result

        # Load front image
        front_file = result.get('front_file', '')
        self.front_viewer.set_image(front_file)

        # Load back image
        back_file = result.get('back_file', '')
        self.back_viewer.set_image(back_file)

        # Update tab text to indicate if back exists
        if back_file and Path(back_file).exists():
            self.image_tabs.setTabText(1, "Back")
        else:
            self.image_tabs.setTabText(1, "Back (none)")

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
