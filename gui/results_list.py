"""
Results List - Tree/table view of processed bills.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLabel, QLineEdit, QComboBox, QPushButton, QMenu, QHeaderView
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QBrush, QAction

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from pattern_engine_v2 import PatternEngine
from settings_manager import get_settings


class NumericTreeWidgetItem(QTreeWidgetItem):
    """TreeWidgetItem that sorts numerically for the first column."""

    def __lt__(self, other):
        column = self.treeWidget().sortColumn() if self.treeWidget() else 0
        if column == 0:  # Position column - sort numerically
            try:
                return int(self.text(0)) < int(other.text(0))
            except ValueError:
                pass
        # Fall back to string comparison (avoid super().__lt__ which can recurse)
        return self.text(column) < other.text(column)


class ResultsList(QWidget):
    """List of processing results with filtering and sorting."""

    # Signals
    item_selected = Signal(dict)  # Emits the selected result
    correction_requested = Signal(dict)  # Emits result needing correction

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results: List[dict] = []
        self.filtered_results: List[dict] = []
        self.filters: Dict[str, bool] = {}
        self.pattern_engine = PatternEngine()
        self.settings = get_settings()
        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Filter bar
        filter_layout = QHBoxLayout()

        # Search box
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search serial...")
        self.search_edit.textChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.search_edit, 1)

        # Pattern filter dropdown
        self.pattern_filter = QComboBox()
        self.pattern_filter.addItem("All Patterns", "")
        self.pattern_filter.setMinimumWidth(120)
        self.pattern_filter.currentIndexChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.pattern_filter)

        # Status filter dropdown
        self.status_filter = QComboBox()
        self.status_filter.addItem("All Status", "all")
        self.status_filter.addItem("Fancy Only", "fancy")
        self.status_filter.addItem("Review Needed", "review")
        self.status_filter.addItem("Errors", "error")
        self.status_filter.currentIndexChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.status_filter)

        layout.addLayout(filter_layout)

        # Results tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["#", "Serial", "Patterns", "Conf", "Status"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setSortingEnabled(True)
        self.tree.setSelectionMode(QTreeWidget.SingleSelection)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)

        # Set column widths
        header = self.tree.header()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)

        layout.addWidget(self.tree)

        # Summary bar
        self.summary_label = QLabel("0 bills")
        layout.addWidget(self.summary_label)

    def add_result(self, result: dict):
        """Add a single result to the list."""
        self.results.append(result)
        self._update_pattern_filter(result)
        self._apply_filters()

    def set_results(self, results: List[dict]):
        """Set all results at once."""
        self.results = results
        self._rebuild_pattern_filter()
        self._apply_filters()

    def clear(self):
        """Clear all results."""
        self.results = []
        self.filtered_results = []
        self.tree.clear()
        self._update_summary()

    def refresh(self):
        """Refresh the display."""
        self._apply_filters()

    def set_filter(self, key: str, enabled: bool):
        """Set a filter flag."""
        self.filters[key] = enabled
        self._apply_filters()

    def _update_pattern_filter(self, result: dict):
        """Update pattern filter dropdown with new patterns."""
        patterns = result.get('fancy_types', '').split(', ')
        for pattern in patterns:
            pattern = pattern.strip()
            if pattern and self.pattern_filter.findText(pattern) == -1:
                self.pattern_filter.addItem(pattern, pattern)

    def _rebuild_pattern_filter(self):
        """Rebuild pattern filter dropdown from all results."""
        self.pattern_filter.clear()
        self.pattern_filter.addItem("All Patterns", "")

        patterns = set()
        for result in self.results:
            for pattern in result.get('fancy_types', '').split(', '):
                pattern = pattern.strip()
                if pattern:
                    patterns.add(pattern)

        for pattern in sorted(patterns):
            self.pattern_filter.addItem(pattern, pattern)

    def _apply_filters(self):
        """Apply all filters and update display."""
        search_text = self.search_edit.text().upper()
        pattern_filter = self.pattern_filter.currentData()
        status_filter = self.status_filter.currentData()

        self.filtered_results = []

        for result in self.results:
            # Search filter
            if search_text:
                serial = result.get('serial', '').upper()
                if search_text not in serial:
                    continue

            # Pattern filter
            if pattern_filter:
                patterns = result.get('fancy_types', '')
                if pattern_filter not in patterns:
                    continue

            # Status filter
            if status_filter == "fancy":
                if not result.get('is_fancy'):
                    continue
            elif status_filter == "review":
                if not result.get('needs_review'):
                    continue
            elif status_filter == "error":
                if not result.get('error'):
                    continue

            # Custom filters from menu
            if self.filters.get('needs_review') and not result.get('needs_review'):
                continue
            if self.filters.get('is_fancy') and not result.get('is_fancy'):
                continue

            self.filtered_results.append(result)

        self._populate_tree()
        self._update_summary()

    def _populate_tree(self):
        """Populate tree with filtered results."""
        self.tree.clear()

        for result in self.filtered_results:
            item = NumericTreeWidgetItem()

            # Position
            item.setText(0, str(result.get('position', '')))
            item.setData(0, Qt.UserRole, result)

            # Serial
            serial = result.get('serial', '')
            if result.get('corrected'):
                serial = f"{serial} (corrected)"
            item.setText(1, serial)

            # Patterns with odds tooltip
            patterns = result.get('fancy_types', '')
            item.setText(2, patterns)

            # Build tooltip with odds for each pattern
            if patterns:
                tooltip_parts = []
                for name in [p.strip() for p in patterns.split(',')]:
                    info = self.pattern_engine.get_pattern_info(name)
                    if info:
                        odds = info.get('odds', 'unknown')
                        desc = info.get('description', '')
                        tooltip_parts.append(f"{name}: {odds}\n  {desc}")
                if tooltip_parts:
                    item.setToolTip(2, '\n\n'.join(tooltip_parts))

            # Confidence
            conf = result.get('confidence', '0.00')
            item.setText(3, str(conf))

            # Status
            status_parts = []
            if result.get('is_fancy'):
                status_parts.append("Fancy")
            if result.get('needs_review'):
                status_parts.append("Review")
            if result.get('error'):
                status_parts.append("Error")
            item.setText(4, ', '.join(status_parts) if status_parts else "OK")

            # Color coding with explicit text color for contrast
            # Tiered color system: Pattern color > Default fancy color
            if result.get('is_fancy'):
                bg_color = None
                pattern_names = [p.strip() for p in patterns.split(',')] if patterns else []

                # Tier 1: Check for pattern-specific custom color
                for pname in pattern_names:
                    custom_color = self.settings.get_pattern_color(pname)
                    if custom_color:
                        bg_color = QColor(custom_color)
                        break  # Use first pattern's custom color

                # Tier 2: Fall back to default fancy color (user-customizable)
                if bg_color is None:
                    default_color = self.settings.ui.default_fancy_color
                    bg_color = QColor(default_color) if default_color else QColor(46, 125, 50)

                for i in range(5):
                    item.setBackground(i, QBrush(bg_color))
                    # Use white or black text based on brightness
                    brightness = (bg_color.red() * 299 + bg_color.green() * 587 + bg_color.blue() * 114) / 1000
                    text_color = QColor(0, 0, 0) if brightness > 128 else QColor(255, 255, 255)
                    item.setForeground(i, QBrush(text_color))
            elif result.get('needs_review'):
                for i in range(5):
                    item.setBackground(i, QBrush(QColor(245, 124, 0)))    # Orange background
                    item.setForeground(i, QBrush(QColor(255, 255, 255)))  # White text
            elif result.get('error'):
                for i in range(5):
                    item.setBackground(i, QBrush(QColor(211, 47, 47)))    # Red background
                    item.setForeground(i, QBrush(QColor(255, 255, 255)))  # White text

            self.tree.addTopLevelItem(item)

    def _update_summary(self):
        """Update summary label."""
        total = len(self.results)
        filtered = len(self.filtered_results)
        fancy = sum(1 for r in self.results if r.get('is_fancy'))
        review = sum(1 for r in self.results if r.get('needs_review'))

        if filtered == total:
            text = f"{total} bills | {fancy} fancy | {review} need review"
        else:
            text = f"{filtered}/{total} bills (filtered) | {fancy} fancy | {review} need review"

        self.summary_label.setText(text)

    def _on_selection_changed(self):
        """Handle selection change."""
        items = self.tree.selectedItems()
        if items:
            result = items[0].data(0, Qt.UserRole)
            self.item_selected.emit(result)

    def _show_context_menu(self, pos):
        """Show context menu for item."""
        item = self.tree.itemAt(pos)
        if not item:
            return

        result = item.data(0, Qt.UserRole)
        menu = QMenu(self)

        # Correct serial action
        correct_action = QAction("Correct Serial...", self)
        correct_action.triggered.connect(lambda: self.correction_requested.emit(result))
        menu.addAction(correct_action)

        # Mark as reviewed
        if result.get('needs_review'):
            mark_reviewed = QAction("Mark as Reviewed", self)
            mark_reviewed.triggered.connect(lambda: self._mark_reviewed(result))
            menu.addAction(mark_reviewed)

        menu.addSeparator()

        # Copy serial
        copy_action = QAction("Copy Serial", self)
        copy_action.triggered.connect(lambda: self._copy_serial(result))
        menu.addAction(copy_action)

        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _mark_reviewed(self, result: dict):
        """Mark an item as reviewed."""
        result['needs_review'] = False
        self._apply_filters()

    def _copy_serial(self, result: dict):
        """Copy serial to clipboard."""
        from PySide6.QtWidgets import QApplication
        serial = result.get('serial', '')
        if serial:
            QApplication.clipboard().setText(serial)

    def get_selected_result(self) -> Optional[dict]:
        """Get currently selected result."""
        items = self.tree.selectedItems()
        if items:
            return items[0].data(0, Qt.UserRole)
        return None

    def select_by_filename(self, filename: str) -> bool:
        """Select an item by its front_file. Returns True if found."""
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            result = item.data(0, Qt.UserRole)
            if result and result.get('front_file') == filename:
                self.tree.setCurrentItem(item)
                return True
        return False

    def select_by_position(self, position: int) -> bool:
        """Select an item by its position. Returns True if found."""
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            result = item.data(0, Qt.UserRole)
            if result and result.get('position') == position:
                self.tree.setCurrentItem(item)
                return True
        return False
