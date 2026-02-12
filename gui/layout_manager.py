"""
Layout Manager - Switchable panel layouts for the main window.

Supports three layout modes:
- classic: Results list on left, Preview + Serial + Details stacked on right
- wide_preview: Preview on top, Results list below (maximizes preview width)
- details_left: Serial + Details on left, Preview + Results stacked on right
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter
)
from PySide6.QtCore import Qt


# Layout identifiers
LAYOUT_CLASSIC = "classic"
LAYOUT_WIDE_PREVIEW = "wide_preview"
LAYOUT_DETAILS_RIGHT = "details_right"

# Human-readable names for menus
LAYOUT_NAMES = {
    LAYOUT_CLASSIC: "Classic",
    LAYOUT_WIDE_PREVIEW: "Wide Preview",
    LAYOUT_DETAILS_RIGHT: "Details Right",
}


class LayoutManager:
    """
    Manages switchable panel layouts by reparenting widgets into different
    QSplitter arrangements.

    The manager does not own the widgets - they are created by MainWindow
    and passed to the manager. The manager handles rearranging them.
    """

    def __init__(self, parent_widget: QWidget):
        """
        Initialize the layout manager.

        Args:
            parent_widget: The widget that will contain the layouts (central widget's layout)
        """
        self.parent_widget = parent_widget
        self.parent_layout = None  # Set during setup

        # Widget references (set by set_widgets)
        self.results_list = None
        self.preview_panel = None
        self.processing_panel = None

        # Current layout container
        self._current_content: QWidget = None
        self._current_layout_name = LAYOUT_CLASSIC

        # Track extracted widgets for details_right layout
        self._details_container = None

        # Keep old containers to avoid deletion issues during reparenting
        self._old_containers = []

    def set_widgets(self, results_list, preview_panel, processing_panel):
        """
        Set the widget references to be managed.

        Args:
            results_list: ResultsList widget
            preview_panel: PreviewPanel widget
            processing_panel: ProcessingPanel widget (not moved, just tracked)
        """
        self.results_list = results_list
        self.preview_panel = preview_panel
        self.processing_panel = processing_panel

    def set_parent_layout(self, layout: QVBoxLayout):
        """Set the parent layout where content will be added."""
        self.parent_layout = layout

    def get_current_layout(self) -> str:
        """Get the current layout name."""
        return self._current_layout_name

    def apply_layout(self, layout_name: str):
        """
        Apply the specified layout.

        Args:
            layout_name: One of LAYOUT_CLASSIC, LAYOUT_WIDE_PREVIEW, LAYOUT_DETAILS_LEFT
        """
        if not self.results_list or not self.preview_panel:
            return

        # First, restore serial/details to preview_panel if they were extracted
        self._restore_preview_panel_widgets()

        # Remove widgets from current parent
        self.results_list.setParent(None)
        self.preview_panel.setParent(None)

        # Remove old content widget from layout
        # Don't delete it immediately - just hide and keep reference to avoid
        # memory corruption during widget reparenting
        if self._current_content:
            self.parent_layout.removeWidget(self._current_content)
            self._current_content.hide()
            self._old_containers.append(self._current_content)
            self._current_content = None

        self._details_container = None

        # Build new layout
        if layout_name == LAYOUT_CLASSIC:
            new_content = self._build_classic_layout()
        elif layout_name == LAYOUT_WIDE_PREVIEW:
            new_content = self._build_wide_preview_layout()
        elif layout_name == LAYOUT_DETAILS_RIGHT:
            new_content = self._build_details_right_layout()
        else:
            # Default to classic
            new_content = self._build_classic_layout()
            layout_name = LAYOUT_CLASSIC

        # Add new content to parent layout (stretch factor 1 for expansion)
        self.parent_layout.addWidget(new_content, 1)
        self._current_content = new_content
        self._current_layout_name = layout_name

    def _build_classic_layout(self) -> QWidget:
        """
        Build classic layout: Results left, Preview+Serial+Details stacked right.

        +-------------------------------------------+
        |         ProcessingPanel (not moved)       |
        +-------------+-----------------------------+
        |             |      Bill Preview           |
        | Results     +-----------------------------+
        |   List      |  Serial Region              |
        |             +-----------------------------+
        |             |  Bill Details               |
        +-------------+-----------------------------+
        """
        splitter = QSplitter(Qt.Horizontal)

        # Left: Results list
        splitter.addWidget(self.results_list)

        # Right: Preview panel (contains serial region and details internally)
        splitter.addWidget(self.preview_panel)

        # Set splitter sizes (40% list, 60% preview)
        splitter.setSizes([400, 600])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        return splitter

    def _build_wide_preview_layout(self) -> QWidget:
        """
        Build wide preview layout: Preview on top, Results below.

        +-------------------------------------------+
        |         ProcessingPanel (not moved)       |
        +-------------------------------------------+
        |          Bill Preview                     |
        |      + Serial Region + Details            |
        +-------------------------------------------+
        |          Results List                     |
        +-------------------------------------------+
        """
        splitter = QSplitter(Qt.Vertical)

        # Top: Preview panel (with serial and details)
        splitter.addWidget(self.preview_panel)

        # Bottom: Results list
        splitter.addWidget(self.results_list)

        # Set splitter sizes (60% preview, 40% results)
        splitter.setSizes([600, 400])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        return splitter

    def _build_details_right_layout(self) -> QWidget:
        """
        Build details-right layout: Preview+Serial on top, Results+Details on bottom.

        +-------------------------------------------+
        |         ProcessingPanel (not moved)       |
        +-------------------------------------------+
        |          Bill Preview                     |
        |          + Serial Region                  |
        +-------------------+-----------------------+
        |   Results List    |    Bill Details       |
        +-------------------+-----------------------+

        The bottom splitter between Results and Details is adjustable.
        """
        main_splitter = QSplitter(Qt.Vertical)

        # Top: Preview panel (keeps serial_frame inside)
        main_splitter.addWidget(self.preview_panel)

        # Bottom: Results + Details side by side
        bottom_splitter = QSplitter(Qt.Horizontal)

        # Left side of bottom: Results list
        bottom_splitter.addWidget(self.results_list)

        # Right side of bottom: Details (reparented from preview_panel)
        self._details_container = QWidget()
        details_layout = QVBoxLayout(self._details_container)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(0)

        if hasattr(self.preview_panel, 'details_group'):
            self.preview_panel.details_group.setParent(None)
            details_layout.addWidget(self.preview_panel.details_group)

        bottom_splitter.addWidget(self._details_container)

        # Set bottom splitter sizes (60% results, 40% details)
        bottom_splitter.setSizes([600, 400])
        bottom_splitter.setStretchFactor(0, 1)
        bottom_splitter.setStretchFactor(1, 0)

        main_splitter.addWidget(bottom_splitter)

        # Set main splitter sizes (60% preview, 40% bottom)
        main_splitter.setSizes([600, 400])
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 0)

        return main_splitter

    def _restore_preview_panel_widgets(self):
        """Restore details_group back into preview_panel's layout."""
        if not self._details_container:
            return

        # Get the preview panel's internal layout
        if not hasattr(self.preview_panel, 'layout'):
            return

        preview_layout = self.preview_panel.layout()
        if not preview_layout:
            return

        # Restore details_group
        if hasattr(self.preview_panel, 'details_group'):
            details_group = self.preview_panel.details_group
            details_group.setParent(None)
            # Add at the end of preview panel layout
            preview_layout.addWidget(details_group)

        self._details_container = None
