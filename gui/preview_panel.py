"""
Preview Panel - Bill image preview and details with zoom/pan support.
"""

import sys
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QPushButton, QLineEdit, QFrame, QGroupBox, QGridLayout,
    QSizePolicy, QTabWidget, QSlider, QApplication, QStackedWidget,
    QComboBox, QSplitter, QMenu, QColorDialog, QCheckBox
)
from PySide6.QtCore import Qt, Signal, Slot, QPoint, QSize
from PySide6.QtGui import QPixmap, QImage, QMouseEvent, QWheelEvent, QCursor, QPainter, QPen, QColor, QAction, QTransform

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pattern_engine_v3 import PatternEngineV3 as PatternEngine

from settings_manager import get_settings


def _load_crosshair_settings():
    """Load crosshair settings from user settings."""
    settings = get_settings()
    color_hex = settings.ui.crosshair_color
    # Convert hex to QColor with alpha
    color = QColor(color_hex)
    color.setAlpha(180)
    return color, settings.ui.crosshair_thickness


class ZoomScrollArea(QScrollArea):
    """QScrollArea that doesn't intercept middle mouse button, allowing child to handle zoom."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._child_widget = None
        self._middle_dragging = False
        self._zoom_start_y = 0

    def setWidget(self, widget):
        """Override to track the child widget."""
        super().setWidget(widget)
        self._child_widget = widget

    def mousePressEvent(self, event):
        """Intercept middle mouse for zoom instead of pan."""
        if event.button() == Qt.MiddleButton and self._child_widget:
            self._middle_dragging = True
            self._zoom_start_y = event.globalPosition().y()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle middle mouse drag for zoom."""
        if self._middle_dragging and self._child_widget:
            current_y = event.globalPosition().y()
            delta_y = current_y - self._zoom_start_y

            # Zoom based on vertical movement
            if abs(delta_y) > 5:
                viewer = getattr(self._child_widget, '_viewer', None)
                if viewer:
                    if delta_y < 0:
                        viewer.zoom_in()
                    else:
                        viewer.zoom_out()
                self._zoom_start_y = current_y
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """End middle mouse zoom drag."""
        if event.button() == Qt.MiddleButton:
            self._middle_dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)


class PannableImageLabel(QLabel):
    """Label that displays an image with pan support via mouse drag and optional crosshair."""

    # Class-level crosshair settings (shared across all instances)
    # These will be initialized from settings on first use
    _crosshair_color = None
    _crosshair_thickness = None
    _settings_loaded = False

    @classmethod
    def _ensure_settings_loaded(cls):
        """Load crosshair settings from user settings if not already loaded."""
        if not cls._settings_loaded:
            cls._crosshair_color, cls._crosshair_thickness = _load_crosshair_settings()
            cls._settings_loaded = True

    def __init__(self, parent=None):
        super().__init__(parent)
        # Ensure crosshair settings are loaded from user settings
        PannableImageLabel._ensure_settings_loaded()
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self._drag_start_global = None  # Use global/screen coordinates
        self._scroll_area = None
        self._viewer = None  # Reference to parent viewer
        self._crosshair_enabled = False
        self._mouse_pos = None  # Track mouse position for crosshair
        self._zoom_start_global = None  # For middle-mouse zoom drag
        self._is_zoom_dragging = False
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_crosshair_menu)

    def set_scroll_area(self, scroll_area: QScrollArea):
        """Set the parent scroll area for panning."""
        self._scroll_area = scroll_area

    def set_viewer(self, viewer):
        """Set the parent viewer to notify about pan state."""
        self._viewer = viewer

    def set_crosshair_enabled(self, enabled: bool):
        """Enable or disable crosshair overlay."""
        self._crosshair_enabled = enabled
        if enabled:
            self.setCursor(QCursor(Qt.CrossCursor))
        else:
            self.setCursor(QCursor(Qt.OpenHandCursor))
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        """Start panning on mouse press, or zooming on middle mouse."""
        if event.button() == Qt.LeftButton and not self._crosshair_enabled:
            # Use global position to avoid jitter from widget movement
            self._drag_start_global = event.globalPosition().toPoint()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            if self._viewer:
                self._viewer._is_panning = True
            super().mousePressEvent(event)
        elif event.button() == Qt.MiddleButton:
            # Middle mouse button for zoom (ThinkPad trackpoint style)
            # Accept event to prevent scroll area from using it for panning
            self._zoom_start_global = event.globalPosition().toPoint()
            self._is_zoom_dragging = True
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Pan the image while dragging, zoom with middle mouse, or update crosshair."""
        # Update mouse position for crosshair
        self._mouse_pos = event.position().toPoint()

        if self._crosshair_enabled:
            self.update()  # Trigger repaint to update crosshair
            super().mouseMoveEvent(event)
        elif self._is_zoom_dragging and self._zoom_start_global and self._viewer:
            # Middle mouse drag for zoom (TrackPoint style)
            current_global = event.globalPosition().toPoint()
            delta_y = current_global.y() - self._zoom_start_global.y()

            # Zoom based on vertical movement: up = zoom in, down = zoom out
            # Use small increments for smooth zooming
            if abs(delta_y) > 5:  # Threshold to avoid tiny movements
                if delta_y < 0:
                    self._viewer.zoom_in()
                else:
                    self._viewer.zoom_out()
                # Reset start position for continuous zooming
                self._zoom_start_global = current_global
            # Accept event to prevent scroll area from panning
            event.accept()
        elif self._drag_start_global and self._scroll_area:
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
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """End panning or zooming on mouse release."""
        if event.button() == Qt.LeftButton and not self._crosshair_enabled:
            self._drag_start_global = None
            self.setCursor(QCursor(Qt.OpenHandCursor))
            if self._viewer:
                self._viewer._is_panning = False
            super().mouseReleaseEvent(event)
        elif event.button() == Qt.MiddleButton:
            self._zoom_start_global = None
            self._is_zoom_dragging = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        """Clear crosshair when mouse leaves."""
        self._mouse_pos = None
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        """Paint the image and optional crosshair."""
        super().paintEvent(event)

        if self._crosshair_enabled and self._mouse_pos is not None:
            painter = QPainter(self)

            # Use class-level crosshair settings
            pen = QPen(PannableImageLabel._crosshair_color)
            pen.setStyle(Qt.DashLine)
            pen.setWidth(PannableImageLabel._crosshair_thickness)
            painter.setPen(pen)

            # Draw vertical line
            painter.drawLine(self._mouse_pos.x(), 0, self._mouse_pos.x(), self.height())
            # Draw horizontal line
            painter.drawLine(0, self._mouse_pos.y(), self.width(), self._mouse_pos.y())

            painter.end()

    def _show_crosshair_menu(self, pos):
        """Show context menu for crosshair settings."""
        if not self._crosshair_enabled:
            return

        menu = QMenu(self)

        # Color submenu
        color_menu = menu.addMenu("Crosshair Color")
        colors = [
            ("Red", QColor(255, 0, 0, 180)),
            ("Green", QColor(0, 255, 0, 180)),
            ("Blue", QColor(0, 100, 255, 180)),
            ("Yellow", QColor(255, 255, 0, 180)),
            ("Cyan", QColor(0, 255, 255, 180)),
            ("Magenta", QColor(255, 0, 255, 180)),
            ("White", QColor(255, 255, 255, 200)),
            ("Black", QColor(0, 0, 0, 200)),
        ]
        for name, color in colors:
            action = color_menu.addAction(name)
            action.setData(color)
            if color.rgb() == PannableImageLabel._crosshair_color.rgb():
                action.setCheckable(True)
                action.setChecked(True)
        color_menu.addSeparator()
        custom_color_action = color_menu.addAction("Custom...")

        # Thickness submenu
        thickness_menu = menu.addMenu("Line Thickness")
        for t in [1, 2, 3, 4, 5]:
            action = thickness_menu.addAction(f"{t}px")
            action.setData(t)
            if t == PannableImageLabel._crosshair_thickness:
                action.setCheckable(True)
                action.setChecked(True)

        # Execute menu
        action = menu.exec(self.mapToGlobal(pos))
        if action:
            if action == custom_color_action:
                color = QColorDialog.getColor(
                    PannableImageLabel._crosshair_color,
                    self,
                    "Select Crosshair Color"
                )
                if color.isValid():
                    color.setAlpha(180)
                    PannableImageLabel._crosshair_color = color
                    self._save_crosshair_settings()
                    self.update()
            elif action.parent() == color_menu:
                PannableImageLabel._crosshair_color = action.data()
                self._save_crosshair_settings()
                self.update()
            elif action.parent() == thickness_menu:
                PannableImageLabel._crosshair_thickness = action.data()
                self._save_crosshair_settings()
                self.update()

    def _save_crosshair_settings(self):
        """Save crosshair settings to user settings."""
        settings = get_settings()
        color = PannableImageLabel._crosshair_color
        # Save as hex (without alpha)
        settings.ui.crosshair_color = color.name()  # Returns #rrggbb
        settings.ui.crosshair_thickness = PannableImageLabel._crosshair_thickness
        settings.save()


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
        self.scroll_area = ZoomScrollArea()
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
        """Handle mouse wheel for zooming (Ctrl+wheel or middle button+trackpoint)."""
        # Zoom with Ctrl+wheel or middle mouse button (ThinkPad trackpoint)
        if event.modifiers() == Qt.ControlModifier or event.buttons() & Qt.MiddleButton:
            delta = event.angleDelta().y()
            # Use smaller zoom steps for smoother trackpoint zooming
            if event.buttons() & Qt.MiddleButton:
                # Trackpoint generates many small events, use gentler zoom
                if delta > 0:
                    self.zoom_factor = min(4.0, self.zoom_factor * 1.05)
                elif delta < 0:
                    self.zoom_factor = max(0.1, self.zoom_factor / 1.05)
                self._update_display()
            else:
                # Regular Ctrl+wheel zoom
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

    def set_crosshair_enabled(self, enabled: bool):
        """Enable or disable crosshair overlay on the image."""
        self.image_label.set_crosshair_enabled(enabled)


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
        self.scroll_area = ZoomScrollArea()
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
        """Forward Ctrl+wheel or middle button+wheel to parent for zoom."""
        if event.modifiers() == Qt.ControlModifier or event.buttons() & Qt.MiddleButton:
            # Let parent handle zoom
            event.ignore()
        else:
            super().wheelEvent(event)

    def set_crosshair_enabled(self, enabled: bool):
        """Enable or disable crosshair overlay."""
        self.image_label.set_crosshair_enabled(enabled)


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
        """Handle Ctrl+wheel or middle button+wheel for zoom."""
        if event.modifiers() == Qt.ControlModifier or event.buttons() & Qt.MiddleButton:
            delta = event.angleDelta().y()
            # Use smaller zoom steps for trackpoint
            if event.buttons() & Qt.MiddleButton:
                if delta > 0:
                    self.zoom_factor = min(4.0, self.zoom_factor * 1.05)
                elif delta < 0:
                    self.zoom_factor = max(0.1, self.zoom_factor / 1.05)
                self._update_zoom()
            else:
                if delta > 0:
                    self._zoom_in()
                else:
                    self._zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)

    def set_crosshair_enabled(self, enabled: bool):
        """Enable or disable crosshair overlay on both panes."""
        self.front_pane.set_crosshair_enabled(enabled)
        self.back_pane.set_crosshair_enabled(enabled)


class ImageLabel(QLabel):
    """Simple label for small images like serial region.
    Left-click: rotate 180°
    Right-click: cycle through pattern overlays
    Right-click hold: show menu
    """

    # Signal emitted when user selects/cycles a pattern
    pattern_selected = Signal(str)  # pattern name or "__all__" or "__gas_pump__" or "__none__"

    # Long press threshold in milliseconds
    LONG_PRESS_MS = 400

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 60)
        self.original_pixmap = None
        self.is_rotated = False
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Left-click: flip 180° | Right-click: cycle overlays | Hold right-click: menu")
        self._matched_patterns = []  # List of (internal_name, display_name) tuples
        self._current_pattern_index = 0  # For cycling: 0=gas_pump, 1+=patterns, last=none

        # For long-press detection
        self._right_press_time = None
        self._left_press_pos = None
        from PySide6.QtCore import QTimer
        self._long_press_timer = QTimer(self)
        self._long_press_timer.setSingleShot(True)
        self._long_press_timer.timeout.connect(self._on_long_press)

    def set_matched_patterns(self, patterns: list):
        """Set the list of matched patterns for context menu.

        Args:
            patterns: List of (internal_name, display_name) tuples
        """
        self._matched_patterns = patterns or []
        self._current_pattern_index = 0  # Reset to gas pump when patterns change

    def _get_cycle_options(self):
        """Get the list of internal pattern names to cycle through."""
        # Order: Gas Pump -> each pattern -> No Overlay -> (back to Gas Pump)
        options = ["__gas_pump__"]
        options.extend([name for name, _ in self._matched_patterns])
        options.append("__none__")
        return options

    def _cycle_next(self):
        """Cycle to the next pattern overlay option."""
        options = self._get_cycle_options()
        self._current_pattern_index = (self._current_pattern_index + 1) % len(options)
        selected = options[self._current_pattern_index]
        self.pattern_selected.emit(selected)

    def _on_long_press(self):
        """Show context menu on long press."""
        if self._left_press_pos is not None:
            self._show_context_menu(self._left_press_pos)
            self._left_press_pos = None  # Prevent cycle on release

    def _show_context_menu(self, pos):
        """Show context menu for pattern overlay selection."""
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)

        # Add "Gas Pump (deviation)" option
        gas_pump_action = menu.addAction("Gas Pump (deviation)")
        gas_pump_action.setData("__gas_pump__")

        # Add separator if we have matched patterns
        if self._matched_patterns:
            menu.addSeparator()

            # Add each matched pattern (show display name, store internal name as data)
            for internal_name, display_name in self._matched_patterns:
                action = menu.addAction(display_name)
                action.setData(internal_name)

        menu.addSeparator()

        # Add "No Overlay" option
        none_action = menu.addAction("No Overlay")
        none_action.setData("__none__")

        # Show menu and handle selection
        action = menu.exec(self.mapToGlobal(pos))
        if action:
            # Update current index to match selection
            options = self._get_cycle_options()
            selected = action.data()
            if selected in options:
                self._current_pattern_index = options.index(selected)
            self.pattern_selected.emit(selected)

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
        self.is_rotated = False  # Reset rotation when new image loaded
        self._update_display()

    def set_pixmap(self, pixmap: QPixmap):
        """Set a pixmap directly."""
        if pixmap is None or pixmap.isNull():
            self.clear()
            return
        self.original_pixmap = pixmap
        self.is_rotated = False  # Reset rotation when new image loaded
        self._update_display()

    def _update_display(self):
        """Update the displayed image scaled to fit."""
        if self.original_pixmap is None:
            return

        pixmap = self.original_pixmap

        # Apply 180° rotation if toggled
        if self.is_rotated:
            transform = QTransform()
            transform.rotate(180)
            pixmap = pixmap.transformed(transform, Qt.SmoothTransformation)

        scaled = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled)

    def mousePressEvent(self, event):
        """Handle click to toggle 180° rotation or start left-click detection for patterns."""
        if event.button() == Qt.RightButton and self.original_pixmap:
            # Right click: rotate 180°
            self.is_rotated = not self.is_rotated
            self._update_display()
        elif event.button() == Qt.LeftButton:
            # Left click: start long press detection for pattern menu
            self._left_press_pos = event.pos()
            self._long_press_timer.start(self.LONG_PRESS_MS)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle left-click release - cycle patterns if it was a quick click."""
        if event.button() == Qt.LeftButton:
            self._long_press_timer.stop()
            # If pos is still set, it was a quick click (not long press)
            if self._left_press_pos is not None:
                self._cycle_next()
                self._left_press_pos = None
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        """Handle resize to update image scaling."""
        super().resizeEvent(event)
        if self.original_pixmap:
            self._update_display()


class PreviewPanel(QWidget):
    """Panel showing bill preview and correction interface."""

    # Signals
    prev_requested = Signal()  # Request to navigate to previous bill
    next_requested = Signal()  # Request to navigate to next bill
    align_requested = Signal(str)  # Request alignment for image path
    px_dev_updated = Signal(int, float)  # (position, fresh_px_dev) - emitted when viewing a bill
    crop_requested = Signal()  # Request to crop the current bill

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
        self._aligned_front_pixmap: Optional[QPixmap] = None  # Cache for aligned front image
        self._aligned_back_pixmap: Optional[QPixmap] = None  # Cache for aligned back image
        self._is_showing_aligned = False
        self._batch_processing_active = False  # Skip heavy ops during batch processing
        self._auto_align_enabled = True  # Auto-align images when selected
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

        header_layout.addSpacing(15)

        # Tool buttons
        self.align_btn = QPushButton("Auto-Align")
        self.align_btn.setToolTip("Toggle auto-alignment using YOLO bill detection")
        self.align_btn.setCheckable(True)
        self.align_btn.setChecked(self._auto_align_enabled)
        self.align_btn.setStyleSheet("""
            QPushButton {
                padding: 4px 8px;
            }
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
        """)
        self.align_btn.clicked.connect(self._on_align_toggled)
        header_layout.addWidget(self.align_btn)

        self.crosshair_btn = QPushButton("Crosshair")
        self.crosshair_btn.setToolTip("Toggle crosshair overlay for alignment checking")
        self.crosshair_btn.setCheckable(True)
        self.crosshair_btn.clicked.connect(self._on_crosshair_toggled)
        header_layout.addWidget(self.crosshair_btn)

        self._crosshair_active = False

        self.crop_btn = QPushButton("Crop (C)")
        self.crop_btn.setToolTip("Generate crops for current bill (C)")
        self.crop_btn.clicked.connect(self.crop_requested.emit)
        header_layout.addWidget(self.crop_btn)

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

        # Serial region images (toggleable via View menu)
        # Shows both serial number regions side by side with bounding boxes
        # Plus a control bar below for color, overlay toggle, and threshold
        self.serial_frame = QFrame()
        serial_main_layout = QVBoxLayout(self.serial_frame)
        serial_main_layout.setContentsMargins(0, 0, 0, 0)
        serial_main_layout.setSpacing(4)

        # Top row: label and two serial images
        serial_images_layout = QHBoxLayout()
        serial_images_layout.setSpacing(8)

        serial_label = QLabel("Serials:")
        serial_images_layout.addWidget(serial_label)

        # Two serial region images (2x zoomed) with pattern mode label between
        self.serial_image_1 = ImageLabel()
        self.serial_image_1.setMinimumSize(300, 80)
        self.serial_image_1.setMaximumHeight(120)
        self.serial_image_1.pattern_selected.connect(self._on_pattern_overlay_selected)
        serial_images_layout.addWidget(self.serial_image_1, 1)

        # Pattern mode label (between the two serial images)
        self.pattern_mode_label = QLabel("Gas Pump")
        self.pattern_mode_label.setAlignment(Qt.AlignCenter)
        self.pattern_mode_label.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 9pt;
                padding: 2px 4px;
                background: #f0f0f0;
                border-radius: 3px;
            }
        """)
        self.pattern_mode_label.setMinimumWidth(80)
        self.pattern_mode_label.setMaximumWidth(120)
        serial_images_layout.addWidget(self.pattern_mode_label)

        self.serial_image_2 = ImageLabel()
        self.serial_image_2.setMinimumSize(300, 80)
        self.serial_image_2.setMaximumHeight(120)
        self.serial_image_2.pattern_selected.connect(self._on_pattern_overlay_selected)
        serial_images_layout.addWidget(self.serial_image_2, 1)

        # Pattern overlay filter: "__gas_pump__", specific pattern, or "__none__"
        self._pattern_overlay_filter = "__gas_pump__"

        serial_main_layout.addLayout(serial_images_layout)

        # Bottom row: control bar
        control_bar = QHBoxLayout()
        control_bar.setSpacing(12)

        # Color picker button with label
        color_label = QLabel("Box Color:")
        control_bar.addWidget(color_label)

        self.bbox_color_btn = QPushButton()
        self.bbox_color_btn.setFixedSize(24, 24)
        self.bbox_color_btn.setToolTip("Click to change bounding box color")
        self.bbox_color_btn.clicked.connect(self._on_bbox_color_clicked)
        self._update_bbox_color_button()
        control_bar.addWidget(self.bbox_color_btn)

        control_bar.addSpacing(20)

        # Pattern overlay checkbox - load saved state from settings
        settings = get_settings()
        self.pattern_overlay_checkbox = QCheckBox("Pattern Overlay")
        self.pattern_overlay_checkbox.setToolTip(
            "Show colored boxes around digits:\n"
            "• Gas pump: green=normal, red=shifted\n"
            "• Flipper: purple outline\n"
            "• Binary: blue outline\n"
            "• Radar: orange pairs\n"
            "• And more..."
        )
        self.pattern_overlay_checkbox.setChecked(settings.ui.gas_pump_overlay_enabled)
        self.pattern_overlay_checkbox.toggled.connect(self._on_pattern_overlay_toggled)
        control_bar.addWidget(self.pattern_overlay_checkbox)

        control_bar.addSpacing(20)

        # Threshold slider with label and value display
        threshold_label = QLabel("Threshold:")
        control_bar.addWidget(threshold_label)

        # Read initial value from pattern config
        initial_threshold = self.pattern_engine.get_gas_pump_threshold()
        self._gas_pump_threshold = initial_threshold

        self.gp_threshold_slider = QSlider(Qt.Horizontal)
        self.gp_threshold_slider.setMinimum(5)   # 0.5 px
        self.gp_threshold_slider.setMaximum(100)  # 10.0 px
        self.gp_threshold_slider.setValue(int(initial_threshold * 10))
        self.gp_threshold_slider.setSingleStep(1)  # Arrow keys: 0.1 px
        self.gp_threshold_slider.setPageStep(2)    # Click on track: 0.2 px
        self.gp_threshold_slider.setMinimumWidth(150)
        self.gp_threshold_slider.setToolTip("Adjust threshold for gas pump detection")
        self.gp_threshold_slider.valueChanged.connect(self._on_gp_threshold_changed)
        control_bar.addWidget(self.gp_threshold_slider)

        self.gp_threshold_value_label = QLabel(f"{initial_threshold:.1f} px")
        self.gp_threshold_value_label.setMinimumWidth(45)
        control_bar.addWidget(self.gp_threshold_value_label)

        control_bar.addStretch()

        self._pattern_overlay_enabled = settings.ui.gas_pump_overlay_enabled
        serial_main_layout.addLayout(control_bar)

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

        # Patterns with Re-classify button
        details_layout.addWidget(QLabel("Patterns:"), 1, 0)
        patterns_layout = QHBoxLayout()
        patterns_layout.setContentsMargins(0, 0, 0, 0)
        patterns_layout.setSpacing(8)
        self.patterns_label = QLabel("-")
        self.patterns_label.setWordWrap(True)
        patterns_layout.addWidget(self.patterns_label, 1)
        self.reclassify_btn = QPushButton("Re-classify")
        self.reclassify_btn.setToolTip("Re-run pattern matching with current patterns (useful after adding new patterns)")
        self.reclassify_btn.setMaximumWidth(80)
        self.reclassify_btn.clicked.connect(self._on_reclassify)
        patterns_layout.addWidget(self.reclassify_btn)
        patterns_widget = QWidget()
        patterns_widget.setLayout(patterns_layout)
        details_layout.addWidget(patterns_widget, 1, 1)

        # Rarity/Odds
        details_layout.addWidget(QLabel("Rarity:"), 2, 0)
        self.odds_label = QLabel("-")
        self.odds_label.setWordWrap(True)
        self.odds_label.setStyleSheet("color: #1976D2; font-weight: bold;")
        details_layout.addWidget(self.odds_label, 2, 1)

        # Price Range
        details_layout.addWidget(QLabel("Est. Price:"), 3, 0)
        self.price_label = QLabel("-")
        self.price_label.setWordWrap(True)
        self.price_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
        details_layout.addWidget(self.price_label, 3, 1)

        # Confidence
        details_layout.addWidget(QLabel("Confidence:"), 4, 0)
        self.confidence_label = QLabel("-")
        details_layout.addWidget(self.confidence_label, 4, 1)

        # Status
        details_layout.addWidget(QLabel("Status:"), 5, 0)
        self.status_label = QLabel("-")
        details_layout.addWidget(self.status_label, 5, 1)

        # File info
        details_layout.addWidget(QLabel("File:"), 6, 0)
        self.file_label = QLabel("-")
        self.file_label.setWordWrap(True)
        details_layout.addWidget(self.file_label, 6, 1)

        layout.addWidget(self.details_group)

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
        # Use aligned images if we're currently showing aligned view
        if self._is_showing_aligned and self._aligned_front_pixmap:
            self._refresh_aligned_views()
        elif mode == "stitched":
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

    def set_batch_processing_active(self, active: bool):
        """Set whether batch processing is active. Skips heavy YOLO ops when True."""
        was_active = self._batch_processing_active
        self._batch_processing_active = active
        # Clear cache when processing ends so next view gets fresh data
        if not active and hasattr(self, '_serial_crop_cache'):
            self._serial_crop_cache.clear()
        # When processing ends, refresh the current result to load images
        if was_active and not active and self.current_result:
            self.show_bill(self.current_result)

    def set_processor(self, processor):
        """Set an existing processor to avoid re-loading YOLO model.

        Call this after processing completes to reuse the already-loaded
        processor for preview operations instead of loading it again.
        """
        self._processor = processor

    def _generate_serial_region_crops(self, image_path: str, zoom: float = 2.0) -> tuple:
        """Generate cropped serial region images with bounding boxes drawn.

        Uses YOLO to detect serial_number boxes on the ALIGNED image and returns
        cropped images with bounding boxes drawn around the detected regions.

        Uses caching to avoid re-running YOLO alignment when just changing colors
        or toggling overlays on the same bill.

        Skips generation during batch processing to keep GUI responsive.

        Args:
            image_path: Path to the front bill image
            zoom: Zoom factor for the crops (default 2.0 for 2x zoom)

        Returns:
            tuple: (list of QPixmap, max_deviation float)
                   max_deviation is the fresh pixel deviation for gas pump detection
        """
        if not image_path or not Path(image_path).exists():
            return [], 0.0

        # Skip heavy YOLO operations during batch processing to keep GUI responsive
        if self._batch_processing_active:
            return [], 0.0

        try:
            # Import processor lazily to avoid circular imports
            from process_production import ProductionProcessor

            # Use cached processor if available, otherwise create one
            if not hasattr(self, '_processor'):
                self._processor = None

            if self._processor is None:
                # Find best.pt model
                model_path = Path(__file__).parent.parent / 'best.pt'
                if model_path.exists():
                    self._processor = ProductionProcessor(str(model_path))
                else:
                    return []

            # Get bounding box color from settings
            settings = get_settings()
            bbox_color_hex = settings.ui.serial_bbox_color
            # Convert hex to BGR for OpenCV
            bbox_color = tuple(int(bbox_color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))

            # Check cache - reuse aligned image and serial boxes if same file
            cache_key = image_path
            if not hasattr(self, '_serial_crop_cache'):
                self._serial_crop_cache = {}

            if cache_key in self._serial_crop_cache:
                # Use cached data
                cached = self._serial_crop_cache[cache_key]
                img = cached['aligned_img']
                serial_boxes = cached['serial_boxes']
            else:
                # Align the image first so serial crops are properly oriented
                aligned_img, info = self._processor.align_for_preview(Path(image_path))
                if aligned_img is None:
                    # Fall back to original if alignment fails
                    img = cv2.imread(image_path)
                    if img is None:
                        return []
                else:
                    img = aligned_img

                # Run YOLO on the aligned image to find serial_number boxes
                results = self._processor.yolo_model(img, verbose=False, conf=0.3)

                serial_boxes = []
                serial_class_id = self._processor.YOLO_CLASSES.get('serial_number', 7)

                for r in results:
                    for box in r.boxes:
                        if hasattr(box, 'cls') and box.cls is not None:
                            cls_id = int(box.cls[0])
                            if cls_id == serial_class_id:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0])
                                serial_boxes.append((x1, y1, x2, y2, conf))

                # Sort by y descending (bottom first) so bottom-left serial shows on left
                serial_boxes.sort(key=lambda b: (-b[1], b[0]))

                # Cache the result (only keep one entry to avoid memory bloat)
                self._serial_crop_cache.clear()
                self._serial_crop_cache[cache_key] = {
                    'aligned_img': img,
                    'serial_boxes': serial_boxes
                }

            if not serial_boxes:
                return [], 0.0

            pixmaps = []
            max_deviation = 0.0
            h, w = img.shape[:2]

            for x1, y1, x2, y2, conf in serial_boxes[:2]:  # Max 2 regions
                # Add padding around the box
                padding = 15
                crop_x1 = max(0, x1 - padding)
                crop_y1 = max(0, y1 - padding)
                crop_x2 = min(w, x2 + padding)
                crop_y2 = min(h, y2 + padding)

                # Crop the region (before zoom, for gas pump analysis)
                crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()

                # Get the tight serial crop (without padding) for gas pump analysis
                tight_crop = img[y1:y2, x1:x2]

                # Always analyze for gas pump to get fresh max_deviation
                gp_result = self._processor.analyze_gas_pump_digits(tight_crop)
                max_deviation = max(max_deviation, gp_result.get('max_deviation', 0.0))

                # Apply zoom before drawing (for sharper bounding box)
                if zoom != 1.0:
                    crop = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)

                # Get overlay filter mode
                overlay_filter = getattr(self, '_pattern_overlay_filter', '__gas_pump__')
                is_gas_pump_mode = (overlay_filter == "__gas_pump__")
                is_pattern_mode = (overlay_filter not in ("__gas_pump__", "__none__"))

                # Draw serial bounding box only in gas pump mode
                if is_gas_pump_mode and self._pattern_overlay_enabled:
                    box_x1 = int((x1 - crop_x1) * zoom)
                    box_y1 = int((y1 - crop_y1) * zoom)
                    box_x2 = int((x2 - crop_x1) * zoom)
                    box_y2 = int((y2 - crop_y1) * zoom)
                    cv2.rectangle(crop, (box_x1, box_y1), (box_x2, box_y2), bbox_color, 2)

                # Draw pattern overlay if enabled
                if self._pattern_overlay_enabled:
                    # Get matched patterns from current result for pattern-based highlighting
                    matched_patterns = []
                    if self.current_result:
                        fancy_types_str = self.current_result.get('fancy_types', '')
                        if fancy_types_str:
                            matched_patterns = [p.strip() for p in fancy_types_str.split(',')]

                    # Get serial for pattern highlights
                    serial = self.current_result.get('serial', '') if self.current_result else ''

                    # Get pattern-based digit highlights for specific pattern mode
                    pattern_highlights = []
                    pattern_connectors = []
                    pattern_group_boxes = []
                    if is_pattern_mode and serial:
                        patterns_for_highlights = [overlay_filter] if overlay_filter in matched_patterns else []
                        if patterns_for_highlights:
                            viz_data = self.pattern_engine.get_digit_highlights(serial, patterns_for_highlights)
                            pattern_highlights = viz_data.get('highlights', [])
                            pattern_connectors = viz_data.get('connectors', [])
                            pattern_group_boxes = viz_data.get('group_boxes', [])

                    # Color map for pattern highlights (CSS name -> BGR)
                    PATTERN_COLORS = {
                        'purple': (128, 0, 128),    # Flipper digits
                        'blue': (255, 0, 0),        # Binary
                        'cyan': (255, 255, 0),      # Trinary
                        'orange': (0, 165, 255),    # Radar pairs
                        'coral': (80, 127, 255),    # Radar pair 2
                        'gold': (0, 215, 255),      # Radar pair 3 / Quads
                        'salmon': (114, 128, 250),  # Radar pair 4
                        'magenta': (255, 0, 255),   # Repeater
                        'yellow': (0, 255, 255),    # Solid/near-solid
                        'lime': (0, 255, 0),        # Ladder
                        'teal': (128, 128, 0),      # Pairs
                        'red': (0, 0, 255),         # Broken/invalid
                        'gray': (128, 128, 128),    # Muted/prefix
                    }

                    # Store digit box info for drawing connectors and group boxes
                    digit_centers = {}  # digit_idx -> (center_x, center_y)
                    digit_rects = {}    # digit_idx -> (x1, y1, x2, y2)

                    # Draw colored boxes for each digit
                    # Sort digit boxes left-to-right to ensure position indices match pattern positions
                    digit_boxes = sorted(gp_result['digit_boxes'], key=lambda db: db['x1'])

                    for idx, digit_box in enumerate(digit_boxes):
                        # Convert digit coordinates to crop-relative, then apply zoom
                        dx1 = int((digit_box['x1'] + (x1 - crop_x1)) * zoom)
                        dy1 = int((digit_box['y1'] + (y1 - crop_y1)) * zoom)
                        dx2 = int((digit_box['x2'] + (x1 - crop_x1)) * zoom)
                        dy2 = int((digit_box['y2'] + (y1 - crop_y1)) * zoom)

                        # Map digit_box index to pattern highlight index (skip letters)
                        digit_idx = sum(1 for db in digit_boxes[:idx] if not db['is_letter'])

                        # Store center and rect for connectors and group boxes
                        if not digit_box['is_letter']:
                            digit_centers[digit_idx] = ((dx1 + dx2) // 2, (dy1 + dy2) // 2)
                            digit_rects[digit_idx] = (dx1, dy1, dx2, dy2)

                        if is_gas_pump_mode:
                            # Gas pump mode: show all boxes with deviation coloring
                            if digit_box['is_letter']:
                                color = (128, 128, 128)  # Gray
                            elif digit_box['deviation'] >= self._gas_pump_threshold:
                                color = (0, 0, 255)  # Red (BGR) - shifted
                            else:
                                color = (0, 255, 0)  # Green (BGR) - normal
                            cv2.rectangle(crop, (dx1, dy1), (dx2, dy2), color, 2)

                        elif is_pattern_mode:
                            # Pattern mode: only show boxes for digits that match the pattern
                            if not digit_box['is_letter'] and digit_idx < len(pattern_highlights):
                                ph = pattern_highlights[digit_idx]
                                if ph['highlights']:
                                    # This digit matches the pattern - show it
                                    first_highlight = ph['highlights'][0]
                                    pattern_color = first_highlight.get('color', 'lime')
                                    color = PATTERN_COLORS.get(pattern_color, (0, 255, 0))
                                    cv2.rectangle(crop, (dx1, dy1), (dx2, dy2), color, 2)

                    # Draw connector lines for relational patterns (e.g., RADAR pairs)
                    if is_pattern_mode and pattern_connectors:
                        for conn in pattern_connectors:
                            # Support both formats: {positions: [a, b]} and {from: a, to: b}
                            if 'positions' in conn:
                                pos1, pos2 = conn['positions']
                            else:
                                pos1 = conn.get('from', 0)
                                pos2 = conn.get('to', 0)

                            if pos1 in digit_centers and pos2 in digit_centers:
                                pt1 = digit_centers[pos1]
                                pt2 = digit_centers[pos2]
                                conn_color = PATTERN_COLORS.get(conn.get('color', 'orange'), (0, 165, 255))
                                conn_style = conn.get('style', 'arc')

                                # Calculate arc - height proportional to distance, but capped
                                mid_x = (pt1[0] + pt2[0]) // 2
                                distance = abs(pt2[0] - pt1[0])
                                # Scale gently and cap at 35px to stay in viewable area
                                arc_height = min(35, max(15, distance // 8))
                                mid_y = min(pt1[1], pt2[1]) - arc_height

                                if conn_style in ('broken', 'dashed'):
                                    # Broken/dashed pair: X marks near each digit (no connecting line)
                                    x_size = 5
                                    # Draw small X near the left digit
                                    x1_pos = pt1[0] + 15  # Offset right from digit center
                                    x1_y = pt1[1] - 10    # Slightly above
                                    cv2.line(crop, (x1_pos - x_size, x1_y - x_size), (x1_pos + x_size, x1_y + x_size), conn_color, 2, cv2.LINE_AA)
                                    cv2.line(crop, (x1_pos - x_size, x1_y + x_size), (x1_pos + x_size, x1_y - x_size), conn_color, 2, cv2.LINE_AA)
                                    # Draw small X near the right digit
                                    x2_pos = pt2[0] - 15  # Offset left from digit center
                                    x2_y = pt2[1] - 10    # Slightly above
                                    cv2.line(crop, (x2_pos - x_size, x2_y - x_size), (x2_pos + x_size, x2_y + x_size), conn_color, 2, cv2.LINE_AA)
                                    cv2.line(crop, (x2_pos - x_size, x2_y + x_size), (x2_pos + x_size, x2_y - x_size), conn_color, 2, cv2.LINE_AA)
                                elif conn_style == 'line':
                                    # Straight line between digit centers
                                    cv2.line(crop, pt1, pt2, conn_color, 2, cv2.LINE_AA)
                                elif conn_style == 'bracket':
                                    # Bracket connector below digits
                                    bracket_y = max(pt1[1], pt2[1]) + 10
                                    cv2.line(crop, (pt1[0], pt1[1] + 5), (pt1[0], bracket_y), conn_color, 2, cv2.LINE_AA)
                                    cv2.line(crop, (pt1[0], bracket_y), (pt2[0], bracket_y), conn_color, 2, cv2.LINE_AA)
                                    cv2.line(crop, (pt2[0], bracket_y), (pt2[0], pt2[1] + 5), conn_color, 2, cv2.LINE_AA)
                                elif conn_style == 'arrow':
                                    # Arrow from left to right
                                    cv2.arrowedLine(crop, pt1, pt2, conn_color, 2, cv2.LINE_AA, tipLength=0.15)
                                else:
                                    # Default arc connector: curved line above digits
                                    pts = np.array([pt1, (mid_x, mid_y), pt2], np.int32)
                                    cv2.polylines(crop, [pts], False, conn_color, 2, cv2.LINE_AA)

                    # Draw group boxes (boxes spanning multiple digits)
                    if is_pattern_mode and pattern_group_boxes:
                        for gb in pattern_group_boxes:
                            # Get start and end positions
                            from_pos = gb.get('from', 0)
                            to_pos = gb.get('to', 0)
                            gb_color = PATTERN_COLORS.get(gb.get('color', 'magenta'), (255, 0, 255))
                            thickness = gb.get('thickness', 3)

                            # Get the bounding rect spanning from first to last digit
                            if from_pos in digit_rects and to_pos in digit_rects:
                                r1 = digit_rects[from_pos]
                                r2 = digit_rects[to_pos]

                                # Combine rects: min x1/y1, max x2/y2 with padding
                                padding = 4
                                gx1 = min(r1[0], r2[0]) - padding
                                gy1 = min(r1[1], r2[1]) - padding
                                gx2 = max(r1[2], r2[2]) + padding
                                gy2 = max(r1[3], r2[3]) + padding

                                # Draw the group box
                                cv2.rectangle(crop, (gx1, gy1), (gx2, gy2), gb_color, thickness)

                # Convert to QPixmap
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                h_crop, w_crop, ch = rgb.shape
                bytes_per_line = ch * w_crop
                qimg = QImage(rgb.data, w_crop, h_crop, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg.copy())  # copy() to own the data
                pixmaps.append(pixmap)

            return pixmaps, max_deviation

        except Exception as e:
            print(f"Error generating serial crops: {e}")
            return [], 0.0

    def _extract_plate_regions(self, zoom: float = 2.0) -> tuple:
        """Extract front and back plate regions at specified zoom for mule comparison.

        Uses YOLO to detect plate regions on aligned front and back images.

        Args:
            zoom: Zoom factor for the extracted regions (default 2.0 for 200%)

        Returns:
            tuple: (front_pixmap, back_pixmap) - either can be None if not detected
        """
        if not self._current_front_file or not Path(self._current_front_file).exists():
            return None, None

        try:
            # Import processor lazily to avoid circular imports
            from process_production import ProductionProcessor

            # Use cached processor if available, otherwise create one
            if not hasattr(self, '_processor'):
                self._processor = None

            if self._processor is None:
                # Find best.pt model
                model_path = Path(__file__).parent.parent / 'best.pt'
                if model_path.exists():
                    self._processor = ProductionProcessor(str(model_path))
                else:
                    return None, None

            front_pixmap = None
            back_pixmap = None

            # Get aligned images
            aligned_front, _ = self._processor.align_for_preview(Path(self._current_front_file))
            if aligned_front is None:
                aligned_front = cv2.imread(self._current_front_file)

            aligned_back = None
            if self._current_back_file and Path(self._current_back_file).exists():
                # For back, we need to align it. Try using cached alignment from current result.
                current_result = self.current_result or {}
                cached_angle = current_result.get('front_align_angle', 0.0)
                cached_flipped = current_result.get('front_align_flipped', False)

                back_img = cv2.imread(self._current_back_file)
                if back_img is not None:
                    h, w = back_img.shape[:2]
                    # Apply same rotation as front (same scan)
                    if abs(cached_angle) >= 0.8:
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, cached_angle, 1.0)
                        back_img = cv2.warpAffine(back_img, M, (w, h),
                                                  flags=cv2.INTER_CUBIC,
                                                  borderMode=cv2.BORDER_CONSTANT,
                                                  borderValue=(255, 255, 255))
                    if cached_flipped:
                        back_img = cv2.rotate(back_img, cv2.ROTATE_180)
                    aligned_back = back_img

            # Extract front plate from front image
            if aligned_front is not None:
                front_detections = self._processor._detect_all_objects(aligned_front, conf=0.3)
                front_plate_boxes = front_detections.get('front_plate', [])

                if front_plate_boxes:
                    # Get best detection
                    best_box = max(front_plate_boxes, key=lambda b: b[4])
                    x1, y1, x2, y2, conf = best_box

                    # Add padding
                    h, w = aligned_front.shape[:2]
                    padding = 10
                    crop_x1 = max(0, x1 - padding)
                    crop_y1 = max(0, y1 - padding)
                    crop_x2 = min(w, x2 + padding)
                    crop_y2 = min(h, y2 + padding)

                    # Crop and zoom
                    crop = aligned_front[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                    if zoom != 1.0:
                        crop = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)

                    # Convert to QPixmap
                    front_pixmap = self._cv2_to_qpixmap(crop)

            # Extract back plate from back image
            if aligned_back is not None:
                back_detections = self._processor._detect_all_objects(aligned_back, conf=0.3)
                back_plate_boxes = back_detections.get('back_plate', [])

                if back_plate_boxes:
                    # Get best detection
                    best_box = max(back_plate_boxes, key=lambda b: b[4])
                    x1, y1, x2, y2, conf = best_box

                    # Add padding
                    h, w = aligned_back.shape[:2]
                    padding = 10
                    crop_x1 = max(0, x1 - padding)
                    crop_y1 = max(0, y1 - padding)
                    crop_x2 = min(w, x2 + padding)
                    crop_y2 = min(h, y2 + padding)

                    # Crop and zoom
                    crop = aligned_back[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                    if zoom != 1.0:
                        crop = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)

                    # Convert to QPixmap
                    back_pixmap = self._cv2_to_qpixmap(crop)

            return front_pixmap, back_pixmap

        except Exception as e:
            print(f"Error extracting plate regions: {e}")
            return None, None

    def _cv2_to_qpixmap(self, cv2_img) -> QPixmap:
        """Convert OpenCV BGR image to QPixmap.

        Args:
            cv2_img: OpenCV image in BGR format

        Returns:
            QPixmap
        """
        h, w, ch = cv2_img.shape
        bytes_per_line = ch * w
        # Convert BGR to RGB for Qt
        rgb_img = cv2_img[:, :, ::-1].copy()
        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)

    def show_plate_magnifier(self):
        """Toggle the plate magnifier popup for mule comparison.

        Extracts front and back plate regions at 200% zoom and displays
        them side-by-side in a popup dialog. Press 'm' again to close.
        """
        # Toggle off if already showing
        if hasattr(self, '_plate_magnifier_dialog') and self._plate_magnifier_dialog is not None:
            if self._plate_magnifier_dialog.isVisible():
                self._plate_magnifier_dialog.close()
                self._plate_magnifier_dialog = None
                return

        if not self.current_result:
            return

        # Check if processor is available
        if not hasattr(self, '_processor') or self._processor is None:
            # Try to create processor lazily
            model_path = Path(__file__).parent.parent / 'best.pt'
            if not model_path.exists():
                # Can't show magnifier without model
                return

        # Extract plate regions
        front_pixmap, back_pixmap = self._extract_plate_regions(zoom=2.0)

        # Show the dialog
        from .plate_magnifier_dialog import PlateMagnifierDialog
        self._plate_magnifier_dialog = PlateMagnifierDialog(front_pixmap, back_pixmap, parent=self)
        # Clear reference when dialog is closed by any means
        self._plate_magnifier_dialog.finished.connect(self._on_plate_magnifier_closed)
        self._plate_magnifier_dialog.show()

    def _on_plate_magnifier_closed(self):
        """Clear plate magnifier dialog reference when closed."""
        self._plate_magnifier_dialog = None

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

    def clear(self):
        """Clear the preview panel, resetting all viewers and labels."""
        # Clear current result
        self.current_result = None
        self._current_front_file = ""
        self._current_back_file = ""
        self._aligned_front_pixmap = None
        self._aligned_back_pixmap = None
        self._is_showing_aligned = False

        # Clear serial crop cache
        if hasattr(self, '_serial_crop_cache'):
            self._serial_crop_cache.clear()

        # Clear all image viewers
        self.front_viewer.set_image("")
        self.back_viewer.set_image("")
        self.combined_viewer.set_pixmap(None)
        self.split_v_viewer.set_images("", "")
        self.split_h_viewer.set_images("", "")
        self.serial_image_1.clear()
        self.serial_image_2.clear()

        # Reset labels
        self.serial_label.setText("-")
        self.patterns_label.setText("-")
        self.odds_label.setText("-")
        self.price_label.setText("-")

        # Clear preserved zoom/pan state
        self.clear_preserved_state()

    def _format_pattern_with_library(self, name: str) -> str:
        """Format a pattern name as 'Display Name (library)'."""
        info = self.pattern_engine.get_pattern_info(name)
        if info:
            display = info.get('display_name', name)
            library = info.get('library', '')
            if library:
                return f"{display} ({library})"
            return display
        return name

    def _format_patterns_display(self, patterns_str: str) -> str:
        """Convert comma-separated pattern names to display names with library."""
        if not patterns_str:
            return "None"
        names = [p.strip() for p in patterns_str.split(',')]
        display_parts = [self._format_pattern_with_library(name) for name in names if name]
        return ', '.join(display_parts) if display_parts else "None"

    def show_bill(self, result: dict):
        """Display a bill result."""
        # Save current zoom/pan state BEFORE loading new images
        has_previous = self.current_result is not None
        if has_previous:
            self._save_zoom_pan_state()

        # Reset aligned state when switching bills
        self._aligned_front_pixmap = None
        self._aligned_back_pixmap = None
        self._is_showing_aligned = False

        self.current_result = result

        # Store file paths for combined view
        self._current_front_file = result.get('front_file', '')
        self._current_back_file = result.get('back_file', '')
        has_back = self._current_back_file and Path(self._current_back_file).exists()

        # Skip expensive image loading during batch processing to avoid slowing down processing
        if self._batch_processing_active:
            # Just update text labels, skip image loading
            self._update_details_only(result, has_back)
            return

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

        # Generate serial region crops on-demand (only if serial view is visible)
        if self.serial_frame.isVisible():
            # Update matched patterns for context menu (list of (internal_name, display_name) tuples)
            fancy_types_str = result.get('fancy_types', '')
            if fancy_types_str:
                matched_patterns = []
                for name in [p.strip() for p in fancy_types_str.split(',')]:
                    if name:
                        matched_patterns.append((name, self._format_pattern_with_library(name)))
            else:
                matched_patterns = []
            self.serial_image_1.set_matched_patterns(matched_patterns)
            self.serial_image_2.set_matched_patterns(matched_patterns)

            serial_crops, fresh_px_dev = self._generate_serial_region_crops(self._current_front_file)
            if len(serial_crops) >= 1:
                self.serial_image_1.set_pixmap(serial_crops[0])
            else:
                self.serial_image_1.clear()
                self.serial_image_1.setText("No serial")

            if len(serial_crops) >= 2:
                self.serial_image_2.set_pixmap(serial_crops[1])
            else:
                self.serial_image_2.clear()
                if len(serial_crops) == 0:
                    self.serial_image_2.setText("detected")
                else:
                    self.serial_image_2.setText("")

            # Note: We no longer emit px_dev_updated here to avoid sorting jumps
            # The processing-time value is kept in the results list

        # Update details
        serial = result.get('serial', '')
        if result.get('corrected'):
            serial += " (corrected)"
        self.serial_label.setText(serial or "-")

        patterns = result.get('fancy_types', '')
        self.patterns_label.setText(self._format_patterns_display(patterns))

        # Look up odds and price for matched patterns
        odds_parts = []
        price_parts = []
        if patterns:
            pattern_names = [p.strip() for p in patterns.split(',')]
            for name in pattern_names:
                info = self.pattern_engine.get_pattern_info(name)
                if info:
                    label = self._format_pattern_with_library(name)
                    if 'odds' in info:
                        odds_parts.append(f"{label}: {info['odds']}")
                    if 'price_range' in info:
                        price_parts.append(f"{label}: {info['price_range']}")
        if odds_parts:
            self.odds_label.setText('\n'.join(odds_parts))
        else:
            self.odds_label.setText("-")
        if price_parts:
            self.price_label.setText('\n'.join(price_parts))
        else:
            self.price_label.setText("-")

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

        self.file_label.setText(self._current_front_file or "-")

        # Update plate magnifier if it's open
        if hasattr(self, '_plate_magnifier_dialog') and self._plate_magnifier_dialog is not None:
            if self._plate_magnifier_dialog.isVisible():
                front_pixmap, back_pixmap = self._extract_plate_regions(zoom=2.0)
                self._plate_magnifier_dialog.update_plates(front_pixmap, back_pixmap)

        # Auto-align if enabled and we have a processor ready
        if self._auto_align_enabled and self._current_front_file:
            # Use a short delay to let the UI update first
            from PySide6.QtCore import QTimer
            QTimer.singleShot(50, lambda: self.align_requested.emit(self._current_front_file))

    def _on_reclassify(self):
        """Re-run pattern classification on the current bill with current patterns."""
        if not self.current_result:
            return

        serial = self.current_result.get('serial', '')
        if not serial:
            return

        # Re-classify using the pattern engine with plate metadata
        metadata = {
            'baseline_variance': float(self.current_result.get('baseline_variance', 0)),
            'series_year': self.current_result.get('series_year', ''),
            'front_plate': self.current_result.get('front_plate', ''),
            'back_plate': self.current_result.get('back_plate', ''),
        }
        matches = self.pattern_engine.classify_simple(serial, metadata)

        # Update the result
        new_fancy_types = ', '.join(matches) if matches else ''
        self.current_result['fancy_types'] = new_fancy_types
        self.current_result['is_fancy'] = len(matches) > 0

        # Update the patterns label
        self.patterns_label.setText(self._format_patterns_display(new_fancy_types))

        # Update odds and price
        odds_parts = []
        price_parts = []
        for name in matches:
            info = self.pattern_engine.get_pattern_info(name)
            if info:
                label = self._format_pattern_with_library(name)
                if info.get('odds'):
                    odds_parts.append(f"{label}: {info['odds']}")
                if info.get('price_range'):
                    price_parts.append(f"{label}: {info['price_range']}")
        self.odds_label.setText('\n'.join(odds_parts) if odds_parts else "-")
        self.price_label.setText('\n'.join(price_parts) if price_parts else "-")

        # Update status
        status_parts = []
        if self.current_result.get('is_fancy'):
            status_parts.append("Fancy")
        if self.current_result.get('needs_review'):
            status_parts.append("Needs Review")
        self.status_label.setText(', '.join(status_parts) if status_parts else "OK")

        # Update matched patterns for overlay cycling (list of (internal_name, display_name) tuples)
        matched_tuples = [(name, self._format_pattern_with_library(name)) for name in matches]
        self.serial_image_1.set_matched_patterns(matched_tuples)
        self.serial_image_2.set_matched_patterns(matched_tuples)

        # Refresh serial region crops to show updated overlays
        if self.serial_frame.isVisible() and self._current_front_file:
            serial_crops, _ = self._generate_serial_region_crops(self._current_front_file)
            if len(serial_crops) >= 1:
                self.serial_image_1.set_pixmap(serial_crops[0])
            if len(serial_crops) >= 2:
                self.serial_image_2.set_pixmap(serial_crops[1])

    def _update_details_only(self, result: dict, has_back: bool):
        """Update only the text details, skip image loading. Used during batch processing."""
        # Update Back button to indicate if back exists
        back_btn = self.view_buttons[1][0]
        if has_back:
            back_btn.setText("Back")
            back_btn.setEnabled(True)
        else:
            back_btn.setText("Back (none)")
            back_btn.setEnabled(False)

        # Update details text
        serial = result.get('serial', '')
        if result.get('corrected'):
            serial += " (corrected)"
        self.serial_label.setText(serial or "-")

        patterns = result.get('fancy_types', '')
        self.patterns_label.setText(self._format_patterns_display(patterns))

        # Look up odds and price for matched patterns
        odds_parts = []
        price_parts = []
        if patterns:
            pattern_names = [p.strip() for p in patterns.split(',')]
            for name in pattern_names:
                info = self.pattern_engine.get_pattern_info(name)
                if info:
                    label = self._format_pattern_with_library(name)
                    if 'odds' in info:
                        odds_parts.append(f"{label}: {info['odds']}")
                    if 'price_range' in info:
                        price_parts.append(f"{label}: {info['price_range']}")
        if odds_parts:
            self.odds_label.setText('\n'.join(odds_parts))
        else:
            self.odds_label.setText("-")
        if price_parts:
            self.price_label.setText('\n'.join(price_parts))
        else:
            self.price_label.setText("-")

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

        self.file_label.setText(self._current_front_file or "-")

        # Show placeholder in image viewers
        self.front_viewer.image_label.setText("Processing... click after completion to view")
        self.serial_image_1.setText("Processing...")
        self.serial_image_2.setText("")

    def set_serial_region_visible(self, visible: bool):
        """Show or hide the serial region panel."""
        self.serial_frame.setVisible(visible)

        # Generate crops when enabling the view (if we have a bill displayed)
        if visible and self._current_front_file:
            serial_crops, fresh_px_dev = self._generate_serial_region_crops(self._current_front_file)
            if len(serial_crops) >= 1:
                self.serial_image_1.set_pixmap(serial_crops[0])
            else:
                self.serial_image_1.clear()
                self.serial_image_1.setText("No serial")

            if len(serial_crops) >= 2:
                self.serial_image_2.set_pixmap(serial_crops[1])
            else:
                self.serial_image_2.clear()
                if len(serial_crops) == 0:
                    self.serial_image_2.setText("detected")
                else:
                    self.serial_image_2.setText("")

            # Note: We no longer emit px_dev_updated to avoid sorting jumps
            # The processing-time value is kept in the results list

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

    def _on_crosshair_toggled(self, checked: bool):
        """Handle crosshair button toggle."""
        self._crosshair_active = checked
        # Apply to all viewers
        for viewer in [self.front_viewer, self.back_viewer, self.combined_viewer,
                       self.split_v_viewer, self.split_h_viewer]:
            if hasattr(viewer, 'set_crosshair_enabled'):
                viewer.set_crosshair_enabled(checked)

    def _on_align_toggled(self, checked: bool):
        """Handle auto-align button toggle."""
        self._auto_align_enabled = checked

        if checked and self._current_front_file:
            # Immediately align the current image when enabling
            self.align_requested.emit(self._current_front_file)
        elif not checked and self._is_showing_aligned:
            # Reset to original when disabling
            self.reset_aligned_image()

    def show_aligned_image(self, pixmap: QPixmap):
        """Display an aligned image in the current viewer (legacy single-image method)."""
        self.show_aligned_images(pixmap, None)

    def _refresh_aligned_views(self):
        """Refresh views using cached aligned images (called when switching view modes)."""
        if not self._aligned_front_pixmap:
            return

        # Use the existing show_aligned_images logic to update all views
        front = self._aligned_front_pixmap
        back = self._aligned_back_pixmap

        # Update the current view mode with aligned images
        mode = self._current_view_mode
        if mode == "front" and front:
            self.front_viewer.set_pixmap(front, preserve_zoom=True)
        elif mode == "back" and back:
            self.back_viewer.set_pixmap(back, preserve_zoom=True)
        elif mode == "stitched":
            combined = self._create_combined_pixmap_from_pixmaps(front, back)
            if combined:
                self.combined_viewer.set_pixmap(combined, preserve_zoom=True)
        elif mode in ("split_v", "split_h"):
            if front:
                self.split_v_viewer.front_pane.original_pixmap = front
                self.split_v_viewer.front_pane._update_display()
                self.split_h_viewer.front_pane.original_pixmap = front
                self.split_h_viewer.front_pane._update_display()
            if back:
                self.split_v_viewer.back_pane.original_pixmap = back
                self.split_v_viewer.back_pane._update_display()
                self.split_h_viewer.back_pane.original_pixmap = back
                self.split_h_viewer.back_pane._update_display()

    def show_aligned_images(self, front_pixmap: QPixmap, back_pixmap: QPixmap):
        """Display aligned images in all viewers."""
        self._is_showing_aligned = True
        self._aligned_front_pixmap = front_pixmap
        self._aligned_back_pixmap = back_pixmap

        # Update front viewer
        if front_pixmap and not front_pixmap.isNull():
            self.front_viewer.set_pixmap(front_pixmap, preserve_zoom=True)

        # Update back viewer
        if back_pixmap and not back_pixmap.isNull():
            self.back_viewer.set_pixmap(back_pixmap, preserve_zoom=True)

        # Update stitched view with aligned images
        if front_pixmap or back_pixmap:
            combined = self._create_combined_pixmap_from_pixmaps(front_pixmap, back_pixmap)
            if combined:
                self.combined_viewer.set_pixmap(combined, preserve_zoom=True)

        # Update split views
        if front_pixmap and back_pixmap:
            self.split_v_viewer.front_pane.original_pixmap = front_pixmap
            self.split_v_viewer.front_pane._update_display()
            self.split_v_viewer.back_pane.original_pixmap = back_pixmap
            self.split_v_viewer.back_pane._update_display()

            self.split_h_viewer.front_pane.original_pixmap = front_pixmap
            self.split_h_viewer.front_pane._update_display()
            self.split_h_viewer.back_pane.original_pixmap = back_pixmap
            self.split_h_viewer.back_pane._update_display()

    def _create_combined_pixmap_from_pixmaps(self, front_pixmap: QPixmap, back_pixmap: QPixmap) -> Optional[QPixmap]:
        """Create a combined pixmap from two pixmaps (for aligned images)."""
        if front_pixmap is None and back_pixmap is None:
            return None
        if front_pixmap is None or front_pixmap.isNull():
            return back_pixmap
        if back_pixmap is None or back_pixmap.isNull():
            return front_pixmap

        # Scale back to match front width
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
        painter.drawPixmap(0, 0, front_pixmap)
        painter.drawPixmap(0, front_pixmap.height(), back_pixmap)
        painter.end()

        return combined

    def reset_aligned_image(self):
        """Reset to original (non-aligned) image."""
        self._aligned_front_pixmap = None
        self._aligned_back_pixmap = None
        self._is_showing_aligned = False

        # Temporarily disable auto-align to prevent re-alignment when reloading
        saved_auto_align = self._auto_align_enabled
        self._auto_align_enabled = False

        # Reload original image
        if self.current_result:
            self.show_bill(self.current_result)

        # Restore auto-align state
        self._auto_align_enabled = saved_auto_align

    def _update_bbox_color_button(self):
        """Update the bounding box color button to show the current color."""
        settings = get_settings()
        color_hex = settings.ui.serial_bbox_color
        self.bbox_color_btn.setStyleSheet(
            f"background-color: {color_hex}; border: 1px solid #555; border-radius: 3px;"
        )

    def _on_bbox_color_clicked(self):
        """Handle bounding box color button click - open color picker."""
        settings = get_settings()
        current_color = QColor(settings.ui.serial_bbox_color)

        color = QColorDialog.getColor(
            current_color,
            self,
            "Select Bounding Box Color"
        )

        if color.isValid():
            # Save the new color
            settings.ui.serial_bbox_color = color.name()
            settings.save()

            # Update the button appearance
            self._update_bbox_color_button()

            # Refresh the serial crops to show the new color
            if self._current_front_file:
                serial_crops, _ = self._generate_serial_region_crops(self._current_front_file)
                if len(serial_crops) >= 1:
                    self.serial_image_1.set_pixmap(serial_crops[0])
                if len(serial_crops) >= 2:
                    self.serial_image_2.set_pixmap(serial_crops[1])

    def _on_pattern_overlay_toggled(self, checked: bool):
        """Handle pattern overlay toggle - show/hide digit boxes."""
        self._pattern_overlay_enabled = checked

        # Save to settings for persistence
        settings = get_settings()
        settings.ui.gas_pump_overlay_enabled = checked  # Reuse same setting key
        settings.save()

        # Refresh the serial crops to show/hide the overlay
        if self._current_front_file and self.serial_frame.isVisible():
            serial_crops, _ = self._generate_serial_region_crops(self._current_front_file)
            if len(serial_crops) >= 1:
                self.serial_image_1.set_pixmap(serial_crops[0])
            if len(serial_crops) >= 2:
                self.serial_image_2.set_pixmap(serial_crops[1])

    def _on_pattern_overlay_selected(self, pattern_filter: str):
        """Handle pattern selection from cycling or context menu."""
        self._pattern_overlay_filter = pattern_filter

        # Update the pattern mode label
        if pattern_filter == "__gas_pump__":
            self.pattern_mode_label.setText("Gas Pump")
            self.pattern_mode_label.setStyleSheet("""
                QLabel { color: #888; font-size: 9pt; padding: 2px 4px; background: #f0f0f0; border-radius: 3px; }
            """)
        elif pattern_filter == "__none__":
            self.pattern_mode_label.setText("No Overlay")
            self.pattern_mode_label.setStyleSheet("""
                QLabel { color: #aaa; font-size: 9pt; padding: 2px 4px; background: #e8e8e8; border-radius: 3px; }
            """)
        else:
            # Specific pattern - show display name with highlight
            info = self.pattern_engine.get_pattern_info(pattern_filter)
            display_name = info.get('display_name', pattern_filter) if info else pattern_filter
            self.pattern_mode_label.setText(display_name)
            self.pattern_mode_label.setStyleSheet("""
                QLabel { color: #2a6; font-size: 9pt; font-weight: bold; padding: 2px 4px; background: #e8f5e9; border-radius: 3px; }
            """)

        # Sync the cycle index between both serial images
        options = self.serial_image_1._get_cycle_options()
        if pattern_filter in options:
            idx = options.index(pattern_filter)
            self.serial_image_1._current_pattern_index = idx
            self.serial_image_2._current_pattern_index = idx

        # Enable overlay if it was disabled and user selected something other than none
        if pattern_filter != "__none__" and not self._pattern_overlay_enabled:
            self._pattern_overlay_enabled = True
            self.pattern_overlay_checkbox.setChecked(True)
        elif pattern_filter == "__none__":
            self._pattern_overlay_enabled = False
            self.pattern_overlay_checkbox.setChecked(False)

        # Refresh the serial crops to show the selected pattern
        if self._current_front_file and self.serial_frame.isVisible():
            serial_crops, _ = self._generate_serial_region_crops(self._current_front_file)
            if len(serial_crops) >= 1:
                self.serial_image_1.set_pixmap(serial_crops[0])
            if len(serial_crops) >= 2:
                self.serial_image_2.set_pixmap(serial_crops[1])

    def _on_gp_threshold_changed(self, value: int):
        """Handle gas pump threshold slider change."""
        # Convert slider value (5-100) to pixels (0.5-10.0)
        self._gas_pump_threshold = value / 10.0
        self.gp_threshold_value_label.setText(f"{self._gas_pump_threshold:.1f} px")

        # Save to pattern config so processing uses the same threshold
        self.pattern_engine.set_gas_pump_threshold(self._gas_pump_threshold)

        # Only refresh if overlay is enabled and we have a bill displayed
        if self._pattern_overlay_enabled and self._current_front_file and self.serial_frame.isVisible():
            serial_crops, _ = self._generate_serial_region_crops(self._current_front_file)
            if len(serial_crops) >= 1:
                self.serial_image_1.set_pixmap(serial_crops[0])
            if len(serial_crops) >= 2:
                self.serial_image_2.set_pixmap(serial_crops[1])
