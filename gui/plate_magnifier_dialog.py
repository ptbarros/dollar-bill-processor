"""
Plate Magnifier Dialog - Popup for comparing front and back plates at 200% zoom.

Shows front and back plate regions side-by-side for mule font-size comparison.
Press 'm' while viewing a bill to show this popup.
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QFont, QKeyEvent

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class PlateMagnifierDialog(QDialog):
    """Popup dialog showing front and back plates at 200% zoom for mule comparison."""

    def __init__(self, front_pixmap: QPixmap = None, back_pixmap: QPixmap = None, parent=None):
        """Initialize the plate magnifier dialog.

        Args:
            front_pixmap: QPixmap of the front plate region (200% zoom), or None if not detected
            back_pixmap: QPixmap of the back plate region (200% zoom), or None if not detected
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)

        self._setup_ui(front_pixmap, back_pixmap)

        # Center over parent
        if parent:
            parent_center = parent.mapToGlobal(parent.rect().center())
            self.move(parent_center.x() - self.width() // 2,
                      parent_center.y() - self.height() // 2)

    def _setup_ui(self, front_pixmap: QPixmap, back_pixmap: QPixmap):
        """Setup the dialog UI."""
        self.setStyleSheet("""
            QDialog {
                background-color: #2d2d2d;
                border: 2px solid #555;
                border-radius: 8px;
            }
            QLabel {
                color: #eee;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Title
        title = QLabel("Plate Comparison")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Plate images side by side
        plates_layout = QHBoxLayout()
        plates_layout.setSpacing(8)

        # Front plate
        front_frame = self._create_plate_frame("Front Plate", front_pixmap)
        plates_layout.addWidget(front_frame)

        # Back plate
        back_frame = self._create_plate_frame("Back Plate", back_pixmap)
        plates_layout.addWidget(back_frame)

        layout.addLayout(plates_layout)

        # Instructions
        instructions = QLabel("Press M or Escape to close")
        instructions.setStyleSheet("color: #888; font-size: 10px;")
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)

    def _create_plate_frame(self, label_text: str, pixmap: QPixmap) -> QFrame:
        """Create a framed plate region with label.

        Args:
            label_text: Header label for the plate
            pixmap: QPixmap of the plate, or None if not detected

        Returns:
            QFrame containing the plate display
        """
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px;
            }
        """)

        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(6, 6, 6, 6)
        frame_layout.setSpacing(2)

        # Header label
        header = QLabel(label_text)
        header.setStyleSheet("font-weight: bold; border: none;")
        header.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(header)

        # Image or placeholder
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumSize(150, 50)
        image_label.setStyleSheet("border: 1px solid #666; background-color: #1d1d1d;")

        if pixmap and not pixmap.isNull():
            image_label.setPixmap(pixmap)
            image_label.setMinimumSize(pixmap.width(), pixmap.height())
        else:
            image_label.setText("Not detected")
            image_label.setStyleSheet(
                "border: 1px solid #666; background-color: #1d1d1d; "
                "color: #888; font-style: italic; padding: 20px;"
            )

        frame_layout.addWidget(image_label)

        return frame

    def keyPressEvent(self, event: QKeyEvent):
        """Close on Escape or M key."""
        if event.key() in (Qt.Key_Escape, Qt.Key_M):
            self.close()
        else:
            super().keyPressEvent(event)
