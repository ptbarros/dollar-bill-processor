"""
Preview Panel - Bill image preview and details with zoom/pan support.
"""

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QPushButton, QLineEdit, QFrame, QGroupBox, QGridLayout,
    QSizePolicy, QTabWidget, QSlider, QApplication, QStackedWidget,
    QComboBox, QSplitter
)
from PySide6.QtCore import Qt, Signal, Slot, QPoint, QSize
from PySide6.QtGui import QPixmap, QImage, QMouseEvent, QWheelEvent, QCursor, QPainter

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from pattern_engine_v2 import PatternEngine


class PannableImageLabel(QLabel):
    """Label that displays an image with pan support via mouse drag."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self._drag_start_global = None  # Use global/screen coordinates
        self._scroll_area = None
        self._viewer = None  # Reference to parent viewer

    def set_scroll_area(self, scroll_area: QScrollArea):
        """Set the parent scroll area for panning."""
        self._scroll_area = scroll_area

    def set_viewer(self, viewer):
        """Set the parent viewer to notify about pan state."""
        self._viewer = viewer

    def mousePressEvent(self, event: QMouseEvent):
        """Start panning on mouse press."""
        if event.button() == Qt.LeftButton:
            # Use global position to avoid jitter from widget movement
            self._drag_start_global = event.globalPosition().toPoint()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            if self._viewer:
                self._viewer._is_panning = True
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Pan the image while dragging."""
        if self._drag_start_global and self._scroll_area:
            # Calculate delta using global coordinates (stable during scroll)
            current_global = event.globalPosition().toPoint()
            delta = current_global - self._drag_start_global

            h_bar = self._scroll_area.horizontalScrollBar()
            v_bar = self._scroll_area.verticalScrollBar()
            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())

            # Update start position for next move
            self._drag_start_global = current_global
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """End panning on mouse release."""
        if event.button() == Qt.LeftButton:
            self._drag_start_global = None
            self.setCursor(QCursor(Qt.OpenHandCursor))
            if self._viewer:
                self._viewer._is_panning = False
        super().mouseReleaseEvent(event)


class ScrollableImageViewer(QWidget):
    """Image viewer with zoom and pan capabilities."""

    zoom_changed = Signal(int)  # Emits zoom percentage

    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = None
        self.zoom_factor = 1.0
        self._is_panning = False  # Track if user is panning
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
        self.image_label.set_viewer(self)  # Connect for pan state tracking
        self.image_label.setCursor(QCursor(Qt.OpenHandCursor))
        self.scroll_area.setWidget(self.image_label)

        layout.addWidget(self.scroll_area, 1)

        # Zoom controls
        zoom_layout = QHBoxLayout()

        self.zoom_fit_btn = QPushButton("Fit")
        self.zoom_fit_btn.clicked.connect(self._zoom_fit)
        zoom_layout.addWidget(self.zoom_fit_btn)

        self.zoom_out_btn = QPushButton("-")
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
        self.zoom_in_btn.clicked.connect(self._zoom_in)
        zoom_layout.addWidget(self.zoom_in_btn)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(45)
        zoom_layout.addWidget(self.zoom_label)

        layout.addLayout(zoom_layout)

    def set_image(self, path: str, preserve_zoom: bool = False):
        """Load and display an image from file path."""
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
        if not preserve_zoom:
            self._zoom_fit()
        else:
            self._update_display()

    def set_pixmap(self, pixmap: QPixmap, preserve_zoom: bool = False):
        """Set a pixmap directly (for combined images)."""
        if pixmap is None or pixmap.isNull():
            self.original_pixmap = None
            self.image_label.clear()
            self.image_label.setText("No image")
            return

        self.original_pixmap = pixmap
        if not preserve_zoom:
            self._zoom_fit()
        else:
            self._update_display()

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
        """Handle resize - refit if at fit zoom and not panning."""
        super().resizeEvent(event)
        # Only auto-fit if we're close to fit zoom and not currently panning
        if self.original_pixmap and self.zoom_factor < 1.0 and not self._is_panning:
            self._zoom_fit()


class ImagePane(QWidget):
    """Simple image pane without zoom controls, for use in synced split view."""

    pan_changed = Signal(float, float)  # Emits scroll position as fraction (0-1)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = None
        self.zoom_factor = 1.0
        self._is_panning = False
        self._syncing = False  # Prevent sync loops
        self._setup_ui()

    def _setup_ui(self):
        """Setup the image pane."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Scroll area for panning (no scrollbars - use drag panning only)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setStyleSheet("QScrollArea { background-color: #2d2d2d; border: none; }")

        # Connect scroll bars for sync
        self.scroll_area.horizontalScrollBar().valueChanged.connect(self._on_scroll_changed)
        self.scroll_area.verticalScrollBar().valueChanged.connect(self._on_scroll_changed)

        # Image label with panning
        self.image_label = PannableImageLabel()
        self.image_label.set_scroll_area(self.scroll_area)
        self.image_label.set_viewer(self)
        self.image_label.setCursor(QCursor(Qt.OpenHandCursor))
        self.scroll_area.setWidget(self.image_label)

        layout.addWidget(self.scroll_area, 1)

    def _on_scroll_changed(self):
        """Emit pan position when scrolling."""
        if self._syncing:
            return

        h_bar = self.scroll_area.horizontalScrollBar()
        v_bar = self.scroll_area.verticalScrollBar()

        # Calculate position as fraction (0-1)
        h_frac = h_bar.value() / max(1, h_bar.maximum()) if h_bar.maximum() > 0 else 0
        v_frac = v_bar.value() / max(1, v_bar.maximum()) if v_bar.maximum() > 0 else 0

        self.pan_changed.emit(h_frac, v_frac)

    def sync_pan_to(self, h_frac: float, v_frac: float):
        """Sync pan position from another pane."""
        self._syncing = True
        h_bar = self.scroll_area.horizontalScrollBar()
        v_bar = self.scroll_area.verticalScrollBar()

        h_bar.setValue(int(h_frac * h_bar.maximum()))
        v_bar.setValue(int(v_frac * v_bar.maximum()))
        self._syncing = False

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
            self.image_label.setText("Failed to load")
            return

        self.original_pixmap = pixmap
        self._update_display()

    def set_zoom(self, factor: float):
        """Set zoom factor."""
        self.zoom_factor = max(0.1, min(4.0, factor))
        self._update_display()

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

    def zoom_fit(self):
        """Fit image to viewport."""
        if self.original_pixmap is None:
            return 1.0

        viewport_size = self.scroll_area.viewport().size()
        img_size = self.original_pixmap.size()

        scale_w = viewport_size.width() / img_size.width()
        scale_h = viewport_size.height() / img_size.height()
        self.zoom_factor = min(scale_w, scale_h) * 0.95

        self._update_display()
        return self.zoom_factor

    def wheelEvent(self, event: QWheelEvent):
        """Forward Ctrl+wheel to parent for zoom."""
        if event.modifiers() == Qt.ControlModifier:
            # Let parent handle zoom
            event.ignore()
        else:
            super().wheelEvent(event)


class SyncedSplitViewer(QWidget):
    """Split view with two synced image panes and shared zoom controls."""

    def __init__(self, orientation: Qt.Orientation = Qt.Vertical, parent=None):
        super().__init__(parent)
        self.orientation = orientation
        self.zoom_factor = 1.0
        self._setup_ui()

    def _setup_ui(self):
        """Setup the split viewer."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Splitter with two image panes - minimal handle for tight spacing
        self.splitter = QSplitter(self.orientation)
        self.splitter.setHandleWidth(4)  # Minimal but still grabbable
        self.splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #555;
            }
            QSplitter::handle:hover {
                background-color: #888;
            }
        """)

        # Front pane (no labels - minimal gap)
        self.front_pane = ImagePane()
        self.front_pane.pan_changed.connect(self._sync_pan_from_front)
        self.splitter.addWidget(self.front_pane)

        # Back pane
        self.back_pane = ImagePane()
        self.back_pane.pan_changed.connect(self._sync_pan_from_back)
        self.splitter.addWidget(self.back_pane)

        # Equal split by default
        self.splitter.setSizes([100, 100])

        layout.addWidget(self.splitter, 1)

        # Shared zoom controls
        zoom_layout = QHBoxLayout()

        self.zoom_fit_btn = QPushButton("Fit")
        self.zoom_fit_btn.clicked.connect(self._zoom_fit)
        zoom_layout.addWidget(self.zoom_fit_btn)

        self.zoom_out_btn = QPushButton("-")
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
        self.zoom_in_btn.clicked.connect(self._zoom_in)
        zoom_layout.addWidget(self.zoom_in_btn)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(45)
        zoom_layout.addWidget(self.zoom_label)

        layout.addLayout(zoom_layout)

    def set_images(self, front_path: str, back_path: str, preserve_zoom: bool = False):
        """Set both images."""
        self.front_pane.set_image(front_path)
        self.back_pane.set_image(back_path)
        if not preserve_zoom:
            self._zoom_fit()
        else:
            self._update_zoom()

    def _sync_pan_from_front(self, h_frac: float, v_frac: float):
        """Sync back pane to front pane's pan position."""
        self.back_pane.sync_pan_to(h_frac, v_frac)

    def _sync_pan_from_back(self, h_frac: float, v_frac: float):
        """Sync front pane to back pane's pan position."""
        self.front_pane.sync_pan_to(h_frac, v_frac)

    def _update_zoom(self):
        """Apply current zoom to both panes."""
        self.front_pane.set_zoom(self.zoom_factor)
        self.back_pane.set_zoom(self.zoom_factor)

        # Update UI
        percent = int(self.zoom_factor * 100)
        self.zoom_label.setText(f"{percent}%")
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(percent)
        self.zoom_slider.blockSignals(False)

    def _zoom_fit(self):
        """Fit both images to their viewports."""
        # Get the smaller fit factor from both panes
        front_factor = self.front_pane.zoom_fit()
        back_factor = self.back_pane.zoom_fit()

        # Use the smaller factor so both fit
        self.zoom_factor = min(front_factor, back_factor) if front_factor and back_factor else 1.0

        self._update_zoom()

    def _zoom_in(self):
        """Zoom in by 25%."""
        self.zoom_factor = min(4.0, self.zoom_factor * 1.25)
        self._update_zoom()

    def _zoom_out(self):
        """Zoom out by 25%."""
        self.zoom_factor = max(0.1, self.zoom_factor / 1.25)
        self._update_zoom()

    def _on_slider_changed(self, value):
        """Handle zoom slider change."""
        self.zoom_factor = value / 100.0
        self._update_zoom()

    def wheelEvent(self, event: QWheelEvent):
        """Handle Ctrl+wheel for zoom."""
        if event.modifiers() == Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self._zoom_in()
            else:
                self._zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)


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
    prev_requested = Signal()  # Request to navigate to previous bill
    next_requested = Signal()  # Request to navigate to next bill

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_result: Optional[dict] = None
        self.pattern_engine = PatternEngine()
        self._current_front_file = ""
        self._current_back_file = ""
        # Preserved zoom/pan state for navigation
        self._preserved_zoom: Optional[float] = None
        self._preserved_scroll_h: Optional[float] = None  # as fraction 0-1
        self._preserved_scroll_v: Optional[float] = None  # as fraction 0-1
        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Image preview area - custom header instead of QGroupBox
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(4)

        # Header row: "Bill Preview" | view buttons | spacer | nav buttons
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)

        title_label = QLabel("Bill Preview")
        title_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(title_label)

        header_layout.addSpacing(10)

        # View mode buttons
        self.view_buttons = []
        view_modes = [
            ("Front", "front", "View front of bill"),
            ("Back", "back", "View back of bill"),
            ("Stitched", "stitched", "View front and back stitched together"),
            ("Split V", "split_v", "View front and back side by side vertically"),
            ("Split H", "split_h", "View front and back side by side horizontally"),
        ]
        for label, mode, tooltip in view_modes:
            btn = QPushButton(label)
            btn.setToolTip(tooltip)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, m=mode: self._on_view_mode_clicked(m))
            header_layout.addWidget(btn)
            self.view_buttons.append((btn, mode))

        # Select "Front" by default
        self.view_buttons[0][0].setChecked(True)
        self._current_view_mode = "front"

        header_layout.addStretch()

        # Navigation buttons
        self.prev_btn = QPushButton("Prev (P)")
        self.prev_btn.setToolTip("Previous bill (Page Up / P)")
        self.prev_btn.clicked.connect(self.prev_requested.emit)
        header_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next (N)")
        self.next_btn.setToolTip("Next bill (Page Down / N)")
        self.next_btn.clicked.connect(self.next_requested.emit)
        header_layout.addWidget(self.next_btn)

        preview_layout.addLayout(header_layout)

        # Stacked widget to hold all view modes
        self.view_stack = QStackedWidget()

        # === Front view (index 0) ===
        self.front_viewer = ScrollableImageViewer()
        self.view_stack.addWidget(self.front_viewer)

        # === Back view (index 1) ===
        self.back_viewer = ScrollableImageViewer()
        self.view_stack.addWidget(self.back_viewer)

        # === Stitched view (index 2) ===
        self.combined_viewer = ScrollableImageViewer()
        self.view_stack.addWidget(self.combined_viewer)

        # === Split Vertical view (index 3) ===
        self.split_v_viewer = SyncedSplitViewer(Qt.Vertical)
        self.view_stack.addWidget(self.split_v_viewer)

        # === Split Horizontal view (index 4) ===
        self.split_h_viewer = SyncedSplitViewer(Qt.Horizontal)
        self.view_stack.addWidget(self.split_h_viewer)

        preview_layout.addWidget(self.view_stack, 1)

        # Serial region image (toggleable via View menu)
        self.serial_frame = QFrame()
        serial_layout = QHBoxLayout(self.serial_frame)
        serial_layout.setContentsMargins(0, 0, 0, 0)

        serial_label = QLabel("Serial Region:")
        serial_layout.addWidget(serial_label)

        self.serial_image = ImageLabel()
        self.serial_image.setMinimumSize(300, 60)
        self.serial_image.setMaximumHeight(80)
        serial_layout.addWidget(self.serial_image, 1)

        preview_layout.addWidget(self.serial_frame)

        # Zoom tip
        zoom_tip = QLabel("Tip: Ctrl+Scroll to zoom, drag to pan")
        zoom_tip.setStyleSheet("color: gray;")
        preview_layout.addWidget(zoom_tip)

        layout.addWidget(preview_container, 1)

        # Details section (toggleable via View menu)
        self.details_group = QGroupBox("Bill Details")
        details_layout = QGridLayout(self.details_group)

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

        layout.addWidget(self.details_group)

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
            btn.clicked.connect(lambda checked, f=from_char, t=to_char: self._apply_quick_fix(f, t))
            quick_layout.addWidget(btn)

        quick_layout.addStretch()
        correction_layout.addLayout(quick_layout)

        layout.addWidget(correction_group)

    def _on_view_mode_clicked(self, mode: str):
        """Handle view mode button click."""
        # Update button states
        for btn, btn_mode in self.view_buttons:
            btn.setChecked(btn_mode == mode)

        self._current_view_mode = mode

        # Map mode to stack index
        mode_to_index = {
            "front": 0,
            "back": 1,
            "stitched": 2,
            "split_v": 3,
            "split_h": 4,
        }
        index = mode_to_index.get(mode, 0)
        self.view_stack.setCurrentIndex(index)

        # Refresh views when switching modes
        if mode == "stitched":
            self._update_combined_view()
        elif mode in ("split_v", "split_h"):
            self._update_split_views()

    def _update_split_views(self, preserve_zoom: bool = False):
        """Update split viewers with current images."""
        self.split_v_viewer.set_images(self._current_front_file, self._current_back_file, preserve_zoom=preserve_zoom)
        self.split_h_viewer.set_images(self._current_front_file, self._current_back_file, preserve_zoom=preserve_zoom)

    def _create_combined_pixmap(self, front_path: str, back_path: str) -> Optional[QPixmap]:
        """Create a combined pixmap with front on top, back on bottom, edge-to-edge."""
        front_pixmap = None
        back_pixmap = None

        # Load front image
        if front_path and Path(front_path).exists():
            front_pixmap = QPixmap(front_path)
            if front_pixmap.isNull():
                front_pixmap = None

        # Load back image
        if back_path and Path(back_path).exists():
            back_pixmap = QPixmap(back_path)
            if back_pixmap.isNull():
                back_pixmap = None

        # Handle cases
        if front_pixmap is None and back_pixmap is None:
            return None
        if front_pixmap is None:
            return back_pixmap
        if back_pixmap is None:
            return front_pixmap

        # Both images exist - combine them
        # Scale back image to match front image width for seamless stitching
        if back_pixmap.width() != front_pixmap.width():
            back_pixmap = back_pixmap.scaledToWidth(
                front_pixmap.width(),
                Qt.SmoothTransformation
            )

        # Create combined image
        combined_height = front_pixmap.height() + back_pixmap.height()
        combined_width = front_pixmap.width()

        combined = QPixmap(combined_width, combined_height)
        combined.fill(Qt.transparent)

        painter = QPainter(combined)
        # Draw front at top
        painter.drawPixmap(0, 0, front_pixmap)
        # Draw back directly below (no gap)
        painter.drawPixmap(0, front_pixmap.height(), back_pixmap)
        painter.end()

        return combined

    def _update_combined_view(self, preserve_zoom: bool = False):
        """Update the combined view with stitched front+back image."""
        combined = self._create_combined_pixmap(
            self._current_front_file,
            self._current_back_file
        )
        if combined:
            self.combined_viewer.set_pixmap(combined, preserve_zoom=preserve_zoom)
        else:
            self.combined_viewer.set_pixmap(None)

    def _save_zoom_pan_state(self):
        """Save the current zoom and pan state from the active viewer."""
        viewer = self._get_active_viewer()
        if viewer is None:
            return

        # Get zoom factor
        if hasattr(viewer, 'zoom_factor'):
            self._preserved_zoom = viewer.zoom_factor
        elif hasattr(viewer, 'front_pane'):
            # Split viewer
            self._preserved_zoom = viewer.zoom_factor

        # Get scroll position as fraction (0-1)
        scroll_area = None
        if hasattr(viewer, 'scroll_area'):
            scroll_area = viewer.scroll_area
        elif hasattr(viewer, 'front_pane'):
            scroll_area = viewer.front_pane.scroll_area

        if scroll_area:
            h_bar = scroll_area.horizontalScrollBar()
            v_bar = scroll_area.verticalScrollBar()
            self._preserved_scroll_h = h_bar.value() / max(1, h_bar.maximum()) if h_bar.maximum() > 0 else 0.5
            self._preserved_scroll_v = v_bar.value() / max(1, v_bar.maximum()) if v_bar.maximum() > 0 else 0.5

    def _restore_zoom_pan_state(self):
        """Restore the saved zoom and pan state to the active viewer."""
        if self._preserved_zoom is None:
            return

        viewer = self._get_active_viewer()
        if viewer is None:
            return

        # Restore zoom
        if hasattr(viewer, 'set_zoom'):
            viewer.set_zoom(self._preserved_zoom)
            viewer._update_display()
        elif hasattr(viewer, '_update_zoom'):
            viewer.zoom_factor = self._preserved_zoom
            viewer._update_zoom()

        # Restore scroll position after a brief delay to ensure layout is complete
        from PySide6.QtCore import QTimer
        QTimer.singleShot(10, self._apply_preserved_scroll)

    def _apply_preserved_scroll(self):
        """Apply preserved scroll position after layout update."""
        if self._preserved_scroll_h is None or self._preserved_scroll_v is None:
            return

        viewer = self._get_active_viewer()
        if viewer is None:
            return

        scroll_area = None
        if hasattr(viewer, 'scroll_area'):
            scroll_area = viewer.scroll_area
        elif hasattr(viewer, 'front_pane'):
            scroll_area = viewer.front_pane.scroll_area

        if scroll_area:
            h_bar = scroll_area.horizontalScrollBar()
            v_bar = scroll_area.verticalScrollBar()
            h_bar.setValue(int(self._preserved_scroll_h * h_bar.maximum()))
            v_bar.setValue(int(self._preserved_scroll_v * v_bar.maximum()))

    def clear_preserved_state(self):
        """Clear the preserved zoom/pan state (e.g., when user resets view)."""
        self._preserved_zoom = None
        self._preserved_scroll_h = None
        self._preserved_scroll_v = None

    def show_bill(self, result: dict):
        """Display a bill result."""
        # Save current zoom/pan state BEFORE loading new images
        has_previous = self.current_result is not None
        if has_previous:
            self._save_zoom_pan_state()

        self.current_result = result

        # Store file paths for combined view
        self._current_front_file = result.get('front_file', '')
        self._current_back_file = result.get('back_file', '')
        has_back = self._current_back_file and Path(self._current_back_file).exists()

        # Determine if we should preserve zoom (only when navigating, not first load)
        preserve = has_previous and self._preserved_zoom is not None

        # Update front and back views
        self.front_viewer.set_image(self._current_front_file, preserve_zoom=preserve)
        self.back_viewer.set_image(self._current_back_file, preserve_zoom=preserve)

        # Update stitched view
        self._update_combined_view(preserve_zoom=preserve)

        # Update split views
        self._update_split_views(preserve_zoom=preserve)

        # Restore zoom/pan state after loading
        if preserve:
            self._restore_zoom_pan_state()

        # Update Back button to indicate if back exists
        back_btn = self.view_buttons[1][0]
        if has_back:
            back_btn.setText("Back")
            back_btn.setEnabled(True)
        else:
            back_btn.setText("Back (none)")
            back_btn.setEnabled(False)

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

    def set_serial_region_visible(self, visible: bool):
        """Show or hide the serial region panel."""
        self.serial_frame.setVisible(visible)

    def set_details_visible(self, visible: bool):
        """Show or hide the bill details panel."""
        self.details_group.setVisible(visible)

    def is_serial_region_visible(self) -> bool:
        """Check if serial region panel is visible."""
        return self.serial_frame.isVisible()

    def is_details_visible(self) -> bool:
        """Check if bill details panel is visible."""
        return self.details_group.isVisible()

    def _get_active_viewer(self):
        """Get the currently active viewer based on view mode."""
        index = self.view_stack.currentIndex()
        if index == 0:  # Front
            return self.front_viewer
        elif index == 1:  # Back
            return self.back_viewer
        elif index == 2:  # Stitched
            return self.combined_viewer
        elif index == 3:  # Split Vertical
            return self.split_v_viewer
        elif index == 4:  # Split Horizontal
            return self.split_h_viewer
        return None

    def zoom_in(self):
        """Zoom in on current view."""
        viewer = self._get_active_viewer()
        if viewer:
            if hasattr(viewer, '_zoom_in'):
                viewer._zoom_in()

    def zoom_out(self):
        """Zoom out on current view."""
        viewer = self._get_active_viewer()
        if viewer:
            if hasattr(viewer, '_zoom_out'):
                viewer._zoom_out()

    def zoom_fit(self):
        """Fit zoom on current view."""
        # Clear preserved state when user explicitly resets to fit
        self.clear_preserved_state()
        viewer = self._get_active_viewer()
        if viewer:
            if hasattr(viewer, '_zoom_fit'):
                viewer._zoom_fit()

    def pan(self, dx: int, dy: int):
        """Pan the current view by given delta."""
        viewer = self._get_active_viewer()
        if viewer is None:
            return

        # For split viewers, pan both panes
        if hasattr(viewer, 'front_pane'):
            h_bar = viewer.front_pane.scroll_area.horizontalScrollBar()
            v_bar = viewer.front_pane.scroll_area.verticalScrollBar()
            h_bar.setValue(h_bar.value() + dx)
            v_bar.setValue(v_bar.value() + dy)
        elif hasattr(viewer, 'scroll_area'):
            h_bar = viewer.scroll_area.horizontalScrollBar()
            v_bar = viewer.scroll_area.verticalScrollBar()
            h_bar.setValue(h_bar.value() + dx)
            v_bar.setValue(v_bar.value() + dy)
