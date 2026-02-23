"""
Results List - Tree/table view of processed bills.
"""

import sys
import csv
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLabel, QLineEdit, QComboBox, QPushButton, QMenu, QHeaderView,
    QInputDialog
)
from PySide6.QtCore import Qt, Signal, Slot, QSettings
from PySide6.QtGui import QColor, QBrush, QAction, QIcon

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pattern_engine_v3 import PatternEngineV3 as PatternEngine

from settings_manager import get_settings
from gui.correction_dialog import CorrectionDialog, ReviewNoteDialog


class NumericTreeWidgetItem(QTreeWidgetItem):
    """TreeWidgetItem that sorts numerically for specific columns."""

    # Columns that should be sorted numerically (by index)
    NUMERIC_COLUMNS = {0, 4, 5, 6, 7}  # Position, GPT, Shift X%, Shift Y%, Seal %

    def __lt__(self, other):
        column = self.treeWidget().sortColumn() if self.treeWidget() else 0
        if column in self.NUMERIC_COLUMNS:
            try:
                # Handle signed numbers like "+5.2" or "-3.1"
                self_val = float(self.text(column).replace('+', ''))
                other_val = float(other.text(column).replace('+', ''))
                return self_val < other_val
            except ValueError:
                pass
        # Fall back to string comparison (avoid super().__lt__ which can recurse)
        return self.text(column) < other.text(column)


class ResultsList(QWidget):
    """List of processing results with filtering and sorting."""

    # Signals
    item_selected = Signal(dict)  # Emits the selected result
    correction_applied = Signal(str, str, str)  # filename, original, corrected
    batch_changed = Signal(str)  # Emits batch path when changed (empty for current session)
    crop_requested = Signal(list)  # Emits list of results to crop
    status_changed = Signal()  # Emits when review status fields change (viewed, cropped, etc.)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results: List[dict] = []
        self.filtered_results: List[dict] = []
        self.filters: Dict[str, bool] = {}
        self.pattern_engine = PatternEngine()
        self.settings = get_settings()
        self._current_batch_path: Optional[Path] = None  # None = current session
        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Batch selector bar
        batch_layout = QHBoxLayout()
        batch_label = QLabel("Batch:")
        batch_layout.addWidget(batch_label)

        self.batch_combo = QComboBox()
        self.batch_combo.addItem("Current Session", "")
        self.batch_combo.setMinimumWidth(200)
        self.batch_combo.currentIndexChanged.connect(self._on_batch_changed)
        batch_layout.addWidget(self.batch_combo, 1)

        self.refresh_batches_btn = QPushButton("Refresh")
        self.refresh_batches_btn.clicked.connect(self.refresh_batch_list)
        batch_layout.addWidget(self.refresh_batches_btn)

        layout.addLayout(batch_layout)

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
        self.status_filter.addItem("Unchecked", "unchecked")
        self.status_filter.addItem("Not Yet Viewed", "unviewed")
        self.status_filter.currentIndexChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.status_filter)

        # Re-classify All button
        self.reclassify_all_btn = QPushButton("Re-classify All")
        self.reclassify_all_btn.setToolTip("Re-run pattern matching on all results (useful after adding new patterns)")
        self.reclassify_all_btn.clicked.connect(self._reclassify_all)
        filter_layout.addWidget(self.reclassify_all_btn)

        # Save CSV button (for saving changes to archived batches)
        self.save_csv_btn = QPushButton("Save CSV")
        self.save_csv_btn.setToolTip("Save current results back to the archive's CSV file")
        self.save_csv_btn.clicked.connect(self._save_csv)
        self.save_csv_btn.setEnabled(False)  # Disabled until an archived batch is selected
        filter_layout.addWidget(self.save_csv_btn)

        layout.addLayout(filter_layout)

        # Results tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["#", "Serial", "Patterns", "Conf", "GPT", "Shift X%", "Shift Y%", "Seal %", "Est. Price", "Series", "Front Plate", "Back Plate", "Mule?", "Mismatch?", "Status"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setSortingEnabled(True)
        self.tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)

        # Column header tooltips explaining what each column means
        self._column_tooltips = {
            0: "Row number in the current list",
            1: "Full serial number (prefix + 8 digits + suffix)",
            2: "Matched fancy serial patterns",
            3: "OCR confidence score (0-100%)",
            4: "Gas Pump Threshold - character baseline variance (lower = more aligned, high = possible gas pump bill)",
            5: "Seal X shift - horizontal offset of treasury seal vs ONE text (%)",
            6: "Seal Y shift - vertical offset of treasury seal vs ONE text (%)",
            7: "Seal containment - % of seal inside ONE bounding box (100% normal, <97% = shifted)",
            8: "Estimated collector price range",
            9: "Bill series year (e.g., 2017A)",
            10: "Front plate number",
            11: "Back plate number",
            12: "Potential mule bill (mismatched front/back plates)",
            13: "Mismatched serial numbers (two different serials detected on front)",
            14: "Status flags: ✓=queued, V=viewed, C=cropped, R=sent for review",
        }
        self._setup_header_tooltips()

        # Set column widths - all interactive for user resizing
        header = self.tree.header()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.Interactive)  # # column
        header.setSectionResizeMode(1, QHeaderView.Interactive)  # Serial
        header.setSectionResizeMode(2, QHeaderView.Interactive)  # Patterns
        header.setSectionResizeMode(3, QHeaderView.Interactive)  # Conf
        header.setSectionResizeMode(4, QHeaderView.Interactive)  # GPT
        header.setSectionResizeMode(5, QHeaderView.Interactive)  # Shift X%
        header.setSectionResizeMode(6, QHeaderView.Interactive)  # Shift Y%
        header.setSectionResizeMode(7, QHeaderView.Interactive)  # Seal %
        header.setSectionResizeMode(8, QHeaderView.Interactive)  # Est. Price
        header.setSectionResizeMode(9, QHeaderView.Interactive)  # Series
        header.setSectionResizeMode(10, QHeaderView.Interactive)  # Front Plate
        header.setSectionResizeMode(11, QHeaderView.Interactive)  # Back Plate
        header.setSectionResizeMode(12, QHeaderView.Interactive)  # Mule?
        header.setSectionResizeMode(13, QHeaderView.Interactive)  # Mismatch?
        header.setSectionResizeMode(14, QHeaderView.Interactive)  # Status
        header.setMinimumSectionSize(30)  # Minimum for any column

        # Move Status column (logical 14) to visual position 1 (between # and Serial)
        header.moveSection(14, 1)

        # Enable right-click context menu on header to hide columns
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(self._show_header_context_menu)

        # Callback for notifying main window when column visibility changes
        self._on_column_visibility_changed = None

        # Load saved column widths or use defaults
        self._load_column_widths()

        # Load saved column visibility
        self._load_column_visibility()

        # Save column widths when user resizes them
        header.sectionResized.connect(self._save_column_widths)

        layout.addWidget(self.tree)

        # Summary bar
        self.summary_label = QLabel("0 bills")
        layout.addWidget(self.summary_label)

    def _load_column_widths(self):
        """Load saved column widths from QSettings, or use defaults."""
        settings = QSettings("DollarBillProcessor", "ResultsList")
        # Default widths: #, Serial, Patterns, Conf, GPT, Shift X%, Shift Y%, Seal %, Est. Price, Series, Front Plate, Back Plate, Mule?, Status
        defaults = [35, 130, 150, 50, 55, 50, 50, 45, 100, 60, 70, 60, 45, 50]

        for i in range(14):
            width = settings.value(f"column_{i}_width", defaults[i], type=int)
            self.tree.setColumnWidth(i, width)

    def _save_column_widths(self, logical_index: int, old_size: int, new_size: int):
        """Save column widths when user resizes them."""
        settings = QSettings("DollarBillProcessor", "ResultsList")
        settings.setValue(f"column_{logical_index}_width", new_size)

    def _setup_header_tooltips(self):
        """Set tooltips on column headers to explain what each column means."""
        header_item = self.tree.headerItem()
        if header_item:
            for col, tooltip in self._column_tooltips.items():
                header_item.setToolTip(col, tooltip)

    def _show_header_context_menu(self, pos):
        """Show context menu when right-clicking on column header."""
        header = self.tree.header()
        logical_index = header.logicalIndexAt(pos)
        if logical_index < 0:
            return

        header_item = self.tree.headerItem()
        col_name = header_item.text(logical_index) if header_item else f"Column {logical_index}"

        menu = QMenu(self)
        hide_action = QAction(f"Hide \"{col_name}\"", self)
        hide_action.triggered.connect(lambda: self._hide_column_from_header(logical_index))
        menu.addAction(hide_action)

        menu.addSeparator()
        restore_hint = QAction("(Use View > Columns to restore)", self)
        restore_hint.setEnabled(False)
        menu.addAction(restore_hint)

        menu.exec(header.mapToGlobal(pos))

    def _hide_column_from_header(self, column: int):
        """Hide a column via header context menu and notify main window."""
        self.set_column_visible(column, False)
        # Notify main window to update its menu checkmarks
        if self._on_column_visibility_changed:
            self._on_column_visibility_changed(column, False)

    def set_column_visibility_callback(self, callback):
        """Set callback to notify when column visibility changes from header menu.

        Args:
            callback: Function(column: int, visible: bool) to call on visibility change
        """
        self._on_column_visibility_changed = callback

    def get_column_info(self) -> list:
        """Return list of (index, name, tooltip, visible) for all columns.

        Used by main window to build the Columns submenu.
        """
        header_item = self.tree.headerItem()
        header = self.tree.header()
        columns = []
        for i in range(self.tree.columnCount()):
            name = header_item.text(i) if header_item else f"Column {i}"
            tooltip = self._column_tooltips.get(i, "")
            visible = not header.isSectionHidden(i)
            columns.append((i, name, tooltip, visible))
        return columns

    def set_column_visible(self, column: int, visible: bool):
        """Show or hide a column by index."""
        self.tree.header().setSectionHidden(column, not visible)
        self._save_column_visibility()

    def is_column_visible(self, column: int) -> bool:
        """Check if a column is visible."""
        return not self.tree.header().isSectionHidden(column)

    def _load_column_visibility(self):
        """Load saved column visibility from QSettings."""
        settings = QSettings("DollarBillProcessor", "ResultsList")
        header = self.tree.header()
        for i in range(self.tree.columnCount()):
            # Default: all columns visible
            hidden = settings.value(f"column_{i}_hidden", False, type=bool)
            header.setSectionHidden(i, hidden)

    def _save_column_visibility(self):
        """Save column visibility to QSettings."""
        settings = QSettings("DollarBillProcessor", "ResultsList")
        header = self.tree.header()
        for i in range(self.tree.columnCount()):
            settings.setValue(f"column_{i}_hidden", header.isSectionHidden(i))

    def _get_display_name(self, pattern_name: str) -> str:
        """Get the display name for a pattern."""
        info = self.pattern_engine.get_pattern_info(pattern_name)
        if info:
            return info.get('display_name', pattern_name)
        return pattern_name

    def _format_patterns_display(self, patterns_str: str) -> str:
        """Convert comma-separated pattern names to display names."""
        if not patterns_str:
            return ''
        names = [p.strip() for p in patterns_str.split(',')]
        display_names = [self._get_display_name(name) for name in names]
        return ', '.join(display_names)

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
            # Check by data value (internal name), not display text
            found = False
            for i in range(self.pattern_filter.count()):
                if self.pattern_filter.itemData(i) == pattern:
                    found = True
                    break
            if pattern and not found:
                display_name = self._get_display_name(pattern)
                self.pattern_filter.addItem(display_name, pattern)

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
            display_name = self._get_display_name(pattern)
            self.pattern_filter.addItem(display_name, pattern)

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
            elif status_filter == "unchecked":
                if result.get('checked'):
                    continue
            elif status_filter == "unviewed":
                if result.get('viewed'):
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
        """Populate tree with filtered results, preserving selection."""
        # Remember current selection and scroll position
        selected_file = None
        selected_item = self.tree.currentItem()
        if selected_item:
            result_data = selected_item.data(0, Qt.UserRole)
            if result_data:
                selected_file = result_data.get('front_file')

        # Remember scroll position
        scrollbar = self.tree.verticalScrollBar()
        scroll_pos = scrollbar.value() if scrollbar else 0

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

            # Patterns - show display names in the column
            patterns = result.get('fancy_types', '')
            item.setText(2, self._format_patterns_display(patterns))

            # Confidence
            conf = result.get('confidence', '0.00')
            item.setText(3, str(conf))

            # Pixel Deviation (for gas pump detection)
            baseline_variance = result.get('baseline_variance', '0.0')
            try:
                px_dev = float(baseline_variance)
                item.setText(4, f"{px_dev:.1f}")
            except (ValueError, TypeError):
                item.setText(4, str(baseline_variance))

            # Overprint shift X (percentage offset)
            seal_x = result.get('seal_x', '0.0')
            try:
                seal_x_val = float(seal_x)
                # Show sign for shift direction (+/- offset)
                item.setText(5, f"{seal_x_val:+.1f}" if seal_x_val != 0 else "0.0")
            except (ValueError, TypeError):
                item.setText(5, str(seal_x))

            # Overprint shift Y (percentage offset, +y = up, -y = down)
            seal_y = result.get('seal_y', '0.0')
            try:
                seal_y_val = float(seal_y)
                # Show sign for shift direction (+/- offset)
                item.setText(6, f"{seal_y_val:+.1f}" if seal_y_val != 0 else "0.0")
            except (ValueError, TypeError):
                item.setText(6, str(seal_y))

            # Seal containment (% of seal inside ONE_hashed bbox)
            seal_containment = result.get('seal_containment', '100.0')
            try:
                seal_cont_val = float(seal_containment)
                item.setText(7, f"{seal_cont_val:.0f}")
            except (ValueError, TypeError):
                item.setText(7, str(seal_containment))

            # Est. Price - get from first matched pattern
            price_text = ""
            if patterns:
                for name in [p.strip() for p in patterns.split(',')]:
                    info = self.pattern_engine.get_pattern_info(name)
                    if info and 'price_range' in info:
                        price_text = info['price_range']
                        break  # Use first pattern's price
            item.setText(8, price_text)

            # Series Year, Front Plate, Back Plate columns
            item.setText(9, result.get('series_year', ''))
            item.setText(10, result.get('front_plate', ''))
            item.setText(11, result.get('back_plate', ''))

            # Mule detection column
            potential_mule = result.get('potential_mule', False)
            if potential_mule:
                item.setText(12, "Yes")
            else:
                item.setText(12, "")

            # Mismatch detection column
            if result.get('serial_mismatch', False):
                item.setText(13, "Yes")
            else:
                item.setText(13, "")

            # Status column (review tracking)
            status_parts = []
            if result.get('checked'):
                status_parts.append('\u2713')
            auto = ''
            if result.get('viewed'):
                auto += 'V'
            if result.get('cropped'):
                auto += 'C'
            if result.get('sent_for_review'):
                auto += 'R'
            if auto:
                status_parts.append(auto)
            item.setText(14, ' '.join(status_parts))

            # Build comprehensive row tooltip with all bill details
            tooltip_lines = [f"Serial: {serial}"]
            if patterns:
                tooltip_lines.append(f"Patterns: {self._format_patterns_display(patterns)}")
                # Add odds for each pattern
                for name in [p.strip() for p in patterns.split(',')]:
                    info = self.pattern_engine.get_pattern_info(name)
                    if info:
                        display_name = info.get('display_name', name)
                        odds = info.get('odds', 'unknown')
                        tooltip_lines.append(f"  {display_name}: {odds}")
            tooltip_lines.append(f"Confidence: {conf}")
            tooltip_lines.append(f"Pixel Dev: {baseline_variance} px (gas pump threshold)")
            tooltip_lines.append(f"Seal shift: X={seal_x}%, Y={seal_y}%, Containment={seal_containment}%")
            if price_text:
                tooltip_lines.append(f"Est. Price: {price_text}")
            # Add filename
            front_file = result.get('front_file', '')
            if front_file:
                tooltip_lines.append(f"File: {Path(front_file).name}")

            row_tooltip = '\n'.join(tooltip_lines)
            for col in range(15):
                item.setToolTip(col, row_tooltip)

            # Color coding with explicit text color for contrast
            # Tiered color system: Pattern color > Library color > Default fancy color
            if result.get('is_fancy'):
                bg_color = None
                pattern_names = [p.strip() for p in patterns.split(',')] if patterns else []

                # Tier 1: Check for pattern-specific custom color
                for pname in pattern_names:
                    custom_color = self.settings.get_pattern_color(pname)
                    if custom_color:
                        bg_color = QColor(custom_color)
                        break  # Use first pattern's custom color

                # Tier 2: Check for library color
                if bg_color is None:
                    for pname in pattern_names:
                        lua_info = self.pattern_engine.lua_patterns.get(pname)
                        if lua_info:
                            lib_color = self.settings.get_library_color(lua_info.library)
                            if lib_color:
                                bg_color = QColor(lib_color)
                                break  # Use first pattern's library color

                # Tier 3: Fall back to default fancy color (user-customizable)
                if bg_color is None:
                    default_color = self.settings.ui.default_fancy_color
                    bg_color = QColor(default_color) if default_color else QColor(46, 125, 50)

                for i in range(12):
                    item.setBackground(i, QBrush(bg_color))
                    # Use white or black text based on brightness
                    brightness = (bg_color.red() * 299 + bg_color.green() * 587 + bg_color.blue() * 114) / 1000
                    text_color = QColor(0, 0, 0) if brightness > 128 else QColor(255, 255, 255)
                    item.setForeground(i, QBrush(text_color))
            elif result.get('needs_review'):
                for i in range(12):
                    item.setBackground(i, QBrush(QColor(245, 124, 0)))    # Orange background
                    item.setForeground(i, QBrush(QColor(255, 255, 255)))  # White text
            elif result.get('error'):
                for i in range(12):
                    item.setBackground(i, QBrush(QColor(211, 47, 47)))    # Red background
                    item.setForeground(i, QBrush(QColor(255, 255, 255)))  # White text

            self.tree.addTopLevelItem(item)

        # Restore selection if the item still exists
        if selected_file:
            for i in range(self.tree.topLevelItemCount()):
                item = self.tree.topLevelItem(i)
                result_data = item.data(0, Qt.UserRole)
                if result_data and result_data.get('front_file') == selected_file:
                    # Block signals to prevent triggering item_selected during restoration
                    self.tree.blockSignals(True)
                    self.tree.setCurrentItem(item)
                    self.tree.blockSignals(False)
                    break

        # Restore scroll position
        if scrollbar and scroll_pos > 0:
            scrollbar.setValue(scroll_pos)

    def _update_summary(self):
        """Update summary label."""
        total = len(self.results)
        filtered = len(self.filtered_results)
        fancy = sum(1 for r in self.results if r.get('is_fancy'))
        review = sum(1 for r in self.results if r.get('needs_review'))
        queued = sum(1 for r in self.results if r.get('checked'))

        if filtered == total:
            text = f"{total} bills | {fancy} fancy | {review} need review"
        else:
            text = f"{filtered}/{total} bills (filtered) | {fancy} fancy | {review} need review"

        if queued:
            text += f" | {queued} queued for crop"

        text += "    Space=queue  C=crop queued"

        self.summary_label.setText(text)

    def _sync_result_field(self, result_copy: dict, field: str, value):
        """Sync a field change back to the authoritative self.results list.

        PySide6's data()/setData() copies dicts, so changes to a dict
        obtained from item.data() won't reflect in self.results.
        """
        front_file = result_copy.get('front_file')
        if front_file:
            for r in self.results:
                if r.get('front_file') == front_file:
                    r[field] = value
                    break

    def _update_status_cell(self, item, result: dict):
        """Update the status column text for a single tree item.

        Note: PySide6's data()/setData() copies dicts, so the caller must
        pass the already-modified result AND store it back via setData().
        """
        status_parts = []
        if result.get('checked'):
            status_parts.append('\u2713')
        auto = ''
        if result.get('viewed'):
            auto += 'V'
        if result.get('cropped'):
            auto += 'C'
        if result.get('sent_for_review'):
            auto += 'R'
        if auto:
            status_parts.append(auto)
        item.setText(14, ' '.join(status_parts))  # Column 14 = Status
        # Store the modified dict back (PySide6 copies on setData)
        item.setData(0, Qt.UserRole, result)

    def _on_selection_changed(self):
        """Handle selection change."""
        items = self.tree.selectedItems()
        if items:
            result = items[0].data(0, Qt.UserRole)
            self.item_selected.emit(result)
            # Auto-track viewed status
            if result and not result.get('viewed'):
                result['viewed'] = True
                self._sync_result_field(result, 'viewed', True)
                sorting_enabled = self.tree.isSortingEnabled()
                self.tree.setSortingEnabled(False)
                self._update_status_cell(items[0], result)
                self.tree.setSortingEnabled(sorting_enabled)

    def _show_context_menu(self, pos):
        """Show context menu for item."""
        item = self.tree.itemAt(pos)
        if not item:
            return

        # If right-clicked item is not in current selection, select only it
        # Otherwise, keep the multi-selection intact
        if not item.isSelected():
            self.tree.setCurrentItem(item)

        # Get all selected items
        selected_items = self.tree.selectedItems()
        selected_results = [i.data(0, Qt.UserRole) for i in selected_items if i.data(0, Qt.UserRole)]
        is_multi_select = len(selected_results) > 1

        # For single-item actions, use the right-clicked item
        result = item.data(0, Qt.UserRole)
        serial = result.get('serial', '')
        menu = QMenu(self)

        # === Single-item actions (only show for single selection) ===
        if not is_multi_select:
            # Correct serial action - opens dialog
            correct_action = QAction("Correct Serial...", self)
            correct_action.triggered.connect(lambda: self._open_correction_dialog(result))
            menu.addAction(correct_action)

            # Quick fixes submenu - position-aware for bill serial format
            # Format: [A-L] + 8 digits + [A-Y or *]
            if serial and len(serial) == 10:
                quick_menu = menu.addMenu("Quick Fixes")
                fixes_added = False

                # Position 0: First letter (must be A-L)
                first_char = serial[0]
                # If digit misread as letter, or letter confusion
                first_pos_fixes = [
                    ("6 → G", "6", "G"),  # 6 misread as G
                    ("8 → B", "8", "B"),  # 8 misread as B
                    ("C → G", "C", "G"),  # C/G confusion
                    ("G → C", "G", "C"),
                ]
                for label, from_char, to_char in first_pos_fixes:
                    if first_char == from_char:
                        action = QAction(f"Pos 1: {label}", self)
                        action.triggered.connect(
                            lambda checked, r=result, pos=0, t=to_char: self._apply_positional_fix(r, pos, t)
                        )
                        quick_menu.addAction(action)
                        fixes_added = True

                # Positions 1-8: Middle digits (must be 0-9)
                # Only offer letter→digit fixes (letters shouldn't be here)
                middle_fixes = [
                    ("O → 0", "O", "0"),
                    ("I → 1", "I", "1"),
                    ("L → 1", "L", "1"),
                    ("S → 5", "S", "5"),
                    ("B → 8", "B", "8"),
                    ("G → 6", "G", "6"),
                    ("Z → 2", "Z", "2"),
                ]
                for idx in range(1, 9):
                    char = serial[idx]
                    for label, from_char, to_char in middle_fixes:
                        if char == from_char:
                            action = QAction(f"Pos {idx+1}: {label}", self)
                            action.triggered.connect(
                                lambda checked, r=result, p=idx, t=to_char: self._apply_positional_fix(r, p, t)
                            )
                            quick_menu.addAction(action)
                            fixes_added = True

                # Position 9: Last letter (must be A-Y or *)
                last_char = serial[9]
                # Digit→letter fixes and letter confusions
                last_pos_fixes = [
                    ("0 → O", "0", "O"),
                    ("0 → Q", "0", "Q"),
                    ("1 → I", "1", "I"),
                    ("1 → L", "1", "L"),
                    ("8 → B", "8", "B"),
                    ("5 → S", "5", "S"),
                    ("2 → Z", "2", "Z"),
                    ("O → Q", "O", "Q"),  # O/Q confusion (both valid)
                    ("Q → O", "Q", "O"),
                    ("C → G", "C", "G"),  # C/G confusion (both valid)
                    ("G → C", "G", "C"),
                ]
                for label, from_char, to_char in last_pos_fixes:
                    if last_char == from_char:
                        action = QAction(f"Pos 10: {label}", self)
                        action.triggered.connect(
                            lambda checked, r=result, pos=9, t=to_char: self._apply_positional_fix(r, pos, t)
                        )
                        quick_menu.addAction(action)
                        fixes_added = True

                if not fixes_added:
                    quick_menu.addAction("(no applicable fixes)").setEnabled(False)

            menu.addSeparator()

        # === Multi-item actions (always show) ===
        # Re-classify selected - re-run pattern matching
        if is_multi_select:
            reclassify_label = f"Re-classify Selected ({len(selected_results)} bills)"
        else:
            reclassify_label = "Re-classify"
        reclassify_action = QAction(reclassify_label, self)
        reclassify_action.setToolTip("Re-run pattern matching (useful after adding new patterns)")
        reclassify_action.triggered.connect(lambda: self._reclassify_selected(selected_results))
        menu.addAction(reclassify_action)

        # === Set Pattern / Set Note (for queue-based workflow) ===
        if not is_multi_select:
            # Single selection - show "Set Pattern..." submenu
            fancy_types = result.get('fancy_types', '')
            patterns = [p.strip() for p in fancy_types.split(',') if p.strip()]
            current_override = result.get('pattern_override', '')

            pattern_menu = menu.addMenu("Set Pattern...")

            # "(Auto)" option to clear override and use first pattern
            auto_action = QAction("(Auto)" if patterns else "(None)", self)
            if not current_override:
                auto_action.setCheckable(True)
                auto_action.setChecked(True)
            auto_action.triggered.connect(lambda: self._set_pattern_override(result, None))
            pattern_menu.addAction(auto_action)

            if patterns:
                pattern_menu.addSeparator()

                # Add each detected pattern
                for pattern in patterns:
                    pattern_action = QAction(pattern, self)
                    pattern_action.setCheckable(True)
                    if current_override == pattern:
                        pattern_action.setChecked(True)
                    pattern_action.triggered.connect(
                        lambda checked, p=pattern: self._set_pattern_override(result, p)
                    )
                    pattern_menu.addAction(pattern_action)

            pattern_menu.addSeparator()

            # "Custom..." option to type a custom pattern name
            custom_action = QAction("Custom...", self)
            if current_override and current_override not in patterns:
                custom_action.setCheckable(True)
                custom_action.setChecked(True)
                custom_action.setText(f"Custom: {current_override}")
            custom_action.triggered.connect(lambda: self._set_custom_pattern(result))
            pattern_menu.addAction(custom_action)

            # "Set Note..." option
            note_action = QAction("Set Note...", self)
            note_action.triggered.connect(lambda: self._set_note(result))
            menu.addAction(note_action)

        menu.addSeparator()

        # === Single-item actions ===
        if not is_multi_select:
            # Save for review
            review_action = QAction("Save for Review...", self)
            review_action.triggered.connect(lambda: self._save_for_review(result))
            menu.addAction(review_action)

            # Mark as reviewed
            if result.get('needs_review'):
                mark_reviewed = QAction("Mark as Reviewed", self)
                mark_reviewed.triggered.connect(lambda: self._mark_reviewed(result))
                menu.addAction(mark_reviewed)

            menu.addSeparator()

        # Toggle checked
        if is_multi_select:
            checked_label = f"Toggle Checked ({len(selected_results)} bills)"
        else:
            checked_label = "Toggle Checked"
        toggle_checked_action = QAction(checked_label, self)
        toggle_checked_action.triggered.connect(self.toggle_checked)
        menu.addAction(toggle_checked_action)

        # Copy serial
        copy_action = QAction("Copy Serial", self)
        copy_action.triggered.connect(lambda: self._copy_serial(result))
        menu.addAction(copy_action)

        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _set_pattern_override(self, result: dict, pattern: Optional[str]):
        """Set or clear the pattern override for a result.

        This stores the override in the result dict for use during crop generation.
        The override is persisted via session recovery.
        """
        front_file = result.get('front_file')
        if not front_file:
            return

        # Update the authoritative results list
        for r in self.results:
            if r.get('front_file') == front_file:
                if pattern:
                    r['pattern_override'] = pattern
                elif 'pattern_override' in r:
                    del r['pattern_override']
                break

        # Update the tree item data
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item_result = item.data(0, Qt.UserRole)
            if item_result and item_result.get('front_file') == front_file:
                if pattern:
                    item_result['pattern_override'] = pattern
                elif 'pattern_override' in item_result:
                    del item_result['pattern_override']
                item.setData(0, Qt.UserRole, item_result)
                break

        # Emit status_changed to trigger autosave
        self.status_changed.emit()

    def _set_custom_pattern(self, result: dict):
        """Open dialog to type a custom pattern name for a result."""
        current_override = result.get('pattern_override', '')
        patterns = [p.strip() for p in result.get('fancy_types', '').split(',') if p.strip()]
        # Pre-fill with current custom override if it's not from detected patterns
        prefill = current_override if current_override and current_override not in patterns else ''

        pattern, ok = QInputDialog.getText(
            self, "Set Custom Pattern",
            "Enter a pattern name for this bill:",
            text=prefill
        )

        if ok and pattern and pattern.strip():
            self._set_pattern_override(result, pattern.strip())

    def _set_note(self, result: dict):
        """Open dialog to set or edit a note for a result."""
        current_note = result.get('note', '')
        note, ok = QInputDialog.getText(
            self, "Set Note",
            "Enter a note for this bill:",
            text=current_note
        )

        if not ok:
            return

        front_file = result.get('front_file')
        if not front_file:
            return

        # Update the authoritative results list
        for r in self.results:
            if r.get('front_file') == front_file:
                if note:
                    r['note'] = note
                elif 'note' in r:
                    del r['note']
                break

        # Update the tree item data
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item_result = item.data(0, Qt.UserRole)
            if item_result and item_result.get('front_file') == front_file:
                if note:
                    item_result['note'] = note
                elif 'note' in item_result:
                    del item_result['note']
                item.setData(0, Qt.UserRole, item_result)
                break

        # Emit status_changed to trigger autosave
        self.status_changed.emit()

    def _open_correction_dialog(self, result: dict):
        """Open the correction dialog for a result."""
        serial = result.get('serial', '')
        filename = result.get('front_file', '')
        image_path = result.get('serial_region_path', '')

        dialog = CorrectionDialog(
            serial=serial,
            image_path=image_path,
            filename=filename,
            parent=self
        )

        if dialog.exec():
            corrected = dialog.get_corrected_serial()
            if corrected and corrected != serial:
                self._apply_correction(result, corrected)

    def _apply_positional_fix(self, result: dict, position: int, to_char: str):
        """Apply a fix at a specific position in the serial."""
        serial = result.get('serial', '')
        if len(serial) > position:
            corrected = serial[:position] + to_char + serial[position + 1:]
            self._apply_correction(result, corrected)

    def _apply_correction(self, result: dict, corrected: str):
        """Apply a correction to a result."""
        filename = result.get('front_file', '')
        original = result.get('serial', '')

        # Find and update the result in self.results (the authoritative source)
        for r in self.results:
            if r.get('front_file') == filename:
                r['serial'] = corrected
                r['corrected'] = True
                break

        # Emit signal for main window to save
        self.correction_applied.emit(filename, original, corrected)

        # Refresh display
        self._apply_filters()

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

    def _save_for_review(self, result: dict):
        """Save a bill to the review folder with a note.

        The review folder is at the project root level, not inside any
        specific batch output. This acts as a universal dev testing tool.
        """
        serial = result.get('serial', '')
        front_file = result.get('front_file', '')
        filename = Path(front_file).name if front_file else 'unknown'

        # Show dialog to get note
        dialog = ReviewNoteDialog(serial=serial, filename=filename, parent=self)
        if not dialog.exec():
            return

        note = dialog.get_note()

        # Universal review folder at project root (not inside batch output)
        project_root = Path(__file__).parent.parent
        review_folder = project_root / "review"
        review_folder.mkdir(exist_ok=True)

        # Copy files to review folder
        files_copied = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Front image
        if front_file and Path(front_file).exists():
            dest = review_folder / f"{timestamp}_{Path(front_file).name}"
            shutil.copy2(front_file, dest)
            files_copied.append(dest.name)

        # Back image
        back_file = result.get('back_file', '')
        if back_file and Path(back_file).exists():
            dest = review_folder / f"{timestamp}_{Path(back_file).name}"
            shutil.copy2(back_file, dest)
            files_copied.append(dest.name)

        # Serial region image
        serial_region = result.get('serial_region_path', '')
        if serial_region and Path(serial_region).exists():
            dest = review_folder / f"{timestamp}_serial_{Path(serial_region).name}"
            shutil.copy2(serial_region, dest)
            files_copied.append(dest.name)

        # Append to CSV log
        csv_path = review_folder / "review_log.csv"
        file_exists = csv_path.exists()

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'serial', 'note', 'confidence',
                                'patterns', 'front_file', 'files_copied'])
            writer.writerow([
                datetime.now().isoformat(),
                serial,
                note,
                result.get('confidence', ''),
                result.get('fancy_types', ''),
                filename,
                '; '.join(files_copied)
            ])

        # Mark as sent for review and update status cell
        result['sent_for_review'] = True
        self._sync_result_field(result, 'sent_for_review', True)
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item_result = item.data(0, Qt.UserRole)
            if item_result and item_result.get('front_file') == result.get('front_file'):
                item_result['sent_for_review'] = True
                self._update_status_cell(item, item_result)
                break
        self.status_changed.emit()

        # Show confirmation
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Saved for Review",
            f"Bill saved to review folder.\n\n"
            f"Files copied: {len(files_copied)}\n"
            f"Note: {note}\n\n"
            f"See: {review_folder}")

    def get_selected_result(self) -> Optional[dict]:
        """Get currently selected result."""
        items = self.tree.selectedItems()
        if items:
            return items[0].data(0, Qt.UserRole)
        return None

    def toggle_checked(self):
        """Toggle checked status on currently selected bill(s)."""
        items = self.tree.selectedItems()
        if not items:
            return
        for item in items:
            result = item.data(0, Qt.UserRole)
            if result:
                new_val = not result.get('checked', False)
                result['checked'] = new_val
                self._sync_result_field(result, 'checked', new_val)
                self._update_status_cell(item, result)
        self._update_summary()

    def mark_cropped(self, results: list):
        """Mark given results as cropped and clear checked flag."""
        cropped_files = {r.get('front_file') for r in results}
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item_result = item.data(0, Qt.UserRole)
            if item_result and item_result.get('front_file') in cropped_files:
                item_result['cropped'] = True
                item_result['checked'] = False  # Clear queue status after crop
                self._sync_result_field(item_result, 'cropped', True)
                self._sync_result_field(item_result, 'checked', False)
                self._update_status_cell(item, item_result)
        self._update_summary()  # Update queued count

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

    # =========================================================================
    # Batch Management
    # =========================================================================

    def refresh_batch_list(self):
        """Scan archive directory and populate batch selector."""
        # Remember current selection
        current_data = self.batch_combo.currentData()

        # Clear and re-add current session
        self.batch_combo.blockSignals(True)
        self.batch_combo.clear()
        self.batch_combo.addItem("Current Session", "")

        # Get archive directory from settings
        archive_dir = self.settings.monitor.archive_directory
        if not archive_dir:
            # Fall back to default location
            archive_dir = str(Path(self.settings.monitor.watch_directory) / "archive")

        archive_path = Path(archive_dir)
        if archive_path.exists():
            # Find all batch directories, sorted newest first
            batch_dirs = sorted(
                [d for d in archive_path.iterdir() if d.is_dir() and d.name.startswith("batch_")],
                key=lambda d: d.name,
                reverse=True
            )

            for batch_dir in batch_dirs:
                # Check if it has a results.csv
                results_csv = batch_dir / "results.csv"
                if results_csv.exists():
                    # Count items in CSV for display
                    try:
                        with open(results_csv, 'r') as f:
                            count = sum(1 for _ in f) - 1  # Subtract header
                        label = f"{batch_dir.name} ({count} bills)"
                    except Exception:
                        label = batch_dir.name
                    self.batch_combo.addItem(label, str(batch_dir))

        # Restore selection if still valid
        idx = self.batch_combo.findData(current_data)
        if idx >= 0:
            self.batch_combo.setCurrentIndex(idx)

        self.batch_combo.blockSignals(False)

    def _on_batch_changed(self, index: int):
        """Handle batch selection change."""
        batch_path = self.batch_combo.currentData()

        if not batch_path:
            # Current session selected
            self._current_batch_path = None
            self.save_csv_btn.setEnabled(False)
            self.batch_changed.emit("")
        else:
            # Archived batch selected
            self._current_batch_path = Path(batch_path)
            self._load_batch(self._current_batch_path)
            self.save_csv_btn.setEnabled(True)
            self.batch_changed.emit(batch_path)

    def _load_batch(self, batch_dir: Path):
        """Load results from an archived batch."""
        results_csv = batch_dir / "results.csv"
        if not results_csv.exists():
            return

        results = []
        try:
            with open(results_csv, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert string booleans to actual booleans
                    result = dict(row)
                    result['is_fancy'] = result.get('is_fancy', '').lower() == 'true'
                    result['needs_review'] = result.get('needs_review', '').lower() == 'true'

                    # Convert position to int
                    try:
                        result['position'] = int(result.get('position', 0))
                    except ValueError:
                        result['position'] = 0

                    # Convert rotation values (for alignment without reprocessing)
                    # Track whether alignment data was present in CSV (vs old archives without it)
                    result['_has_alignment_data'] = 'front_align_angle' in row and row['front_align_angle'] != ''
                    try:
                        result['front_align_angle'] = float(result.get('front_align_angle', 0.0))
                    except (ValueError, TypeError):
                        result['front_align_angle'] = 0.0
                    result['front_align_flipped'] = result.get('front_align_flipped', '').lower() == 'true'

                    # Ensure plate info fields exist (backward compatibility with older CSVs)
                    result['series_year'] = result.get('series_year', '')
                    result['front_plate'] = result.get('front_plate', '')
                    result['back_plate'] = result.get('back_plate', '')
                    result['potential_mule'] = result.get('potential_mule', '').lower() == 'true'
                    result['serial_mismatch'] = result.get('serial_mismatch', '').lower() == 'true'

                    # Review status fields (backward compatible - missing columns default to False)
                    result['viewed'] = result.get('viewed', '').lower() == 'true'
                    result['cropped'] = result.get('cropped', '').lower() == 'true'
                    result['sent_for_review'] = result.get('sent_for_review', '').lower() == 'true'
                    result['checked'] = result.get('checked', '').lower() == 'true'

                    # User fields (backward compatible - missing columns default to empty)
                    note = result.get('note', '')
                    if note:
                        result['note'] = note
                    elif 'note' in result:
                        del result['note']

                    pattern_override = result.get('pattern_override', '')
                    if pattern_override:
                        result['pattern_override'] = pattern_override
                    elif 'pattern_override' in result:
                        del result['pattern_override']

                    # Update file paths to point to archive location
                    front_file = result.get('front_file', '')
                    if front_file:
                        # Use just the filename and look in batch dir
                        front_name = Path(front_file).name
                        archived_path = batch_dir / front_name
                        if archived_path.exists():
                            result['front_file'] = str(archived_path)

                    back_file = result.get('back_file', '')
                    if back_file:
                        back_name = Path(back_file).name
                        archived_path = batch_dir / back_name
                        if archived_path.exists():
                            result['back_file'] = str(archived_path)

                    results.append(result)

        except Exception as e:
            print(f"Error loading batch: {e}")
            return

        # Set results (this will update the display)
        self.set_results(results)

    def get_current_batch_path(self) -> Optional[Path]:
        """Get the path of the currently selected batch, or None for current session."""
        return self._current_batch_path

    def _save_csv(self):
        """Save current results back to the archive's CSV file."""
        if not self._current_batch_path:
            return

        csv_path = self._current_batch_path / "results.csv"
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'position', 'front_file', 'back_file', 'serial', 'fancy_types',
                    'confidence', 'baseline_variance', 'seal_x', 'seal_y', 'seal_containment',
                    'is_fancy', 'needs_review', 'serial_region_path', 'error',
                    'front_align_angle', 'front_align_flipped',
                    'series_year', 'front_plate', 'back_plate', 'potential_mule', 'serial_mismatch',
                    'viewed', 'cropped', 'sent_for_review', 'checked',
                    'note', 'pattern_override'
                ])
                writer.writeheader()

                for result in self.results:
                    # Create a clean copy for CSV output (exclude internal fields like _has_alignment_data)
                    row = {k: v for k, v in result.items() if not k.startswith('_')}
                    # Convert paths back to just filenames for portability
                    if 'front_file' in row and row['front_file']:
                        row['front_file'] = Path(row['front_file']).name
                    if 'back_file' in row and row['back_file']:
                        row['back_file'] = Path(row['back_file']).name
                    if 'serial_region_path' in row and row['serial_region_path']:
                        row['serial_region_path'] = Path(row['serial_region_path']).name
                    writer.writerow(row)

            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "CSV Saved",
                f"Results saved to:\n{csv_path}")

        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Save Error",
                f"Failed to save CSV:\n{e}")

    def update_px_dev(self, position: int, px_dev: float):
        """Update the GPT (Gas Pump Threshold) column for a specific result by position.

        Called when viewing a bill to show the fresh calculated deviation
        instead of the value from processing time.
        """
        # Temporarily disable sorting to prevent the item from jumping
        # when the value changes while sorted by this column
        sorting_enabled = self.tree.isSortingEnabled()
        self.tree.setSortingEnabled(False)

        # Find the tree item with this position
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item and item.text(0) == str(position):
                item.setText(4, f"{px_dev:.1f}")
                # Also update the underlying result data
                for result in self.results:
                    if result.get('position') == position:
                        result['baseline_variance'] = f"{px_dev:.1f}"
                        break
                break

        # Re-enable sorting (but don't trigger a re-sort)
        self.tree.setSortingEnabled(sorting_enabled)

    def update_result_paths(self, path_mapping: dict):
        """Update file paths in results after archiving.

        Args:
            path_mapping: Dict mapping old paths to new paths
        """
        # Update paths in internal results list
        for result in self.results:
            front_file = result.get('front_file', '')
            back_file = result.get('back_file', '')
            if front_file and front_file in path_mapping:
                result['front_file'] = path_mapping[front_file]
            if back_file and back_file in path_mapping:
                result['back_file'] = path_mapping[back_file]

        # Update paths in tree items' UserRole data
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item:
                result = item.data(0, Qt.UserRole)
                if result:
                    front_file = result.get('front_file', '')
                    back_file = result.get('back_file', '')
                    if front_file and front_file in path_mapping:
                        result['front_file'] = path_mapping[front_file]
                    if back_file and back_file in path_mapping:
                        result['back_file'] = path_mapping[back_file]
                    item.setData(0, Qt.UserRole, result)

    def select_current_session(self):
        """Switch back to current session view."""
        self.batch_combo.setCurrentIndex(0)

    # =========================================================================
    # Re-classification
    # =========================================================================

    def _reclassify_selected(self, results: list):
        """Re-run pattern matching on selected results."""
        if not results:
            return

        # Reload patterns to pick up any new ones
        self.pattern_engine.reload()

        for result in results:
            self._reclassify_result(result)

        # Refresh the display
        self._apply_filters()

        # Re-select to update preview panel
        self._on_selection_changed()

    def _reclassify_all(self):
        """Re-run pattern matching on all results."""
        if not self.results:
            return

        # Reload patterns to pick up any new ones
        self.pattern_engine.reload()

        for result in self.results:
            self._reclassify_result(result)

        # Refresh the display
        self._apply_filters()

        # Re-select to update preview panel
        self._on_selection_changed()

    def _reclassify_result(self, result: dict):
        """Re-classify a single result and update its data."""
        serial = result.get('serial', '')
        if not serial:
            return

        # Re-run pattern matching with plate metadata
        metadata = {
            'baseline_variance': float(result.get('baseline_variance', 0)),
            'gas_pump_threshold': self.pattern_engine.get_gas_pump_threshold(),
            'series_year': result.get('series_year', ''),
            'front_plate': result.get('front_plate', ''),
            'back_plate': result.get('back_plate', ''),
        }
        matches = self.pattern_engine.classify_simple(serial, metadata)

        # Update the result
        new_fancy_types = ', '.join(matches) if matches else ''
        result['fancy_types'] = new_fancy_types
        result['is_fancy'] = len(matches) > 0

        # Update the tree item if it exists
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item_result = item.data(0, Qt.UserRole)
            if item_result and item_result.get('front_file') == result.get('front_file'):
                # Update the Patterns column (column 2)
                item.setText(2, new_fancy_types or "-")

                # Update colors based on fancy status
                if result.get('is_fancy'):
                    item.setForeground(2, QBrush(QColor("#2e7d32")))  # Green for fancy
                else:
                    item.setForeground(2, QBrush(QColor("#000000")))  # Black for normal

                # Update the stored data
                item.setData(0, Qt.UserRole, result)
                break
