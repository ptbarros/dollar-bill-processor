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
    QLabel, QLineEdit, QComboBox, QPushButton, QMenu, QHeaderView
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QBrush, QAction

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from pattern_engine_v2 import PatternEngine
from settings_manager import get_settings
from gui.correction_dialog import CorrectionDialog, ReviewNoteDialog


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
    correction_applied = Signal(str, str, str)  # filename, original, corrected
    batch_changed = Signal(str)  # Emits batch path when changed (empty for current session)

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
        self.refresh_batches_btn.setMaximumWidth(60)
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
        self.status_filter.currentIndexChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.status_filter)

        layout.addLayout(filter_layout)

        # Results tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["#", "Serial", "Patterns", "Conf", "Px Dev", "Est. Price"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setSortingEnabled(True)
        self.tree.setSelectionMode(QTreeWidget.SingleSelection)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)

        # Set column widths - all interactive for user resizing
        header = self.tree.header()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.Interactive)  # # column
        header.setSectionResizeMode(1, QHeaderView.Interactive)  # Serial
        header.setSectionResizeMode(2, QHeaderView.Stretch)      # Patterns (takes remaining space)
        header.setSectionResizeMode(3, QHeaderView.Interactive)  # Conf
        header.setSectionResizeMode(4, QHeaderView.Interactive)  # Px Dev
        header.setSectionResizeMode(5, QHeaderView.Interactive)  # Est. Price

        # Set minimum and default widths
        self.tree.setColumnWidth(0, 35)   # # column
        self.tree.setColumnWidth(1, 130)  # Serial - enough for full serial at font 14
        self.tree.setColumnWidth(3, 50)   # Conf
        self.tree.setColumnWidth(4, 55)   # Px Dev
        self.tree.setColumnWidth(5, 100)  # Est. Price
        header.setMinimumSectionSize(30)  # Minimum for any column

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

            # Patterns
            patterns = result.get('fancy_types', '')
            item.setText(2, patterns)

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

            # Est. Price - get from first matched pattern
            price_text = ""
            if patterns:
                for name in [p.strip() for p in patterns.split(',')]:
                    info = self.pattern_engine.get_pattern_info(name)
                    if info and 'price_range' in info:
                        price_text = info['price_range']
                        break  # Use first pattern's price
            item.setText(5, price_text)

            # Build comprehensive row tooltip with all bill details
            tooltip_lines = [f"Serial: {serial}"]
            if patterns:
                tooltip_lines.append(f"Patterns: {patterns}")
                # Add odds for each pattern
                for name in [p.strip() for p in patterns.split(',')]:
                    info = self.pattern_engine.get_pattern_info(name)
                    if info:
                        odds = info.get('odds', 'unknown')
                        tooltip_lines.append(f"  {name}: {odds}")
            tooltip_lines.append(f"Confidence: {conf}")
            tooltip_lines.append(f"Pixel Dev: {baseline_variance} px (gas pump threshold)")
            if price_text:
                tooltip_lines.append(f"Est. Price: {price_text}")
            # Add filename
            front_file = result.get('front_file', '')
            if front_file:
                tooltip_lines.append(f"File: {Path(front_file).name}")

            row_tooltip = '\n'.join(tooltip_lines)
            for col in range(6):
                item.setToolTip(col, row_tooltip)

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

        # Select the item when right-clicking (ensures consistency)
        self.tree.setCurrentItem(item)

        result = item.data(0, Qt.UserRole)
        serial = result.get('serial', '')
        menu = QMenu(self)

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

        # Copy serial
        copy_action = QAction("Copy Serial", self)
        copy_action.triggered.connect(lambda: self._copy_serial(result))
        menu.addAction(copy_action)

        menu.exec(self.tree.viewport().mapToGlobal(pos))

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
            self.batch_changed.emit("")
        else:
            # Archived batch selected
            self._current_batch_path = Path(batch_path)
            self._load_batch(self._current_batch_path)
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

    def update_px_dev(self, position: int, px_dev: float):
        """Update the Px Dev column for a specific result by position.

        Called when viewing a bill to show the fresh calculated deviation
        instead of the value from processing time.
        """
        # Find the tree item with this position
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item and item.text(0) == str(position):
                item.setText(4, f"{px_dev:.1f}")
                # Also update the underlying result data
                for result in self._all_results:
                    if result.get('position') == position:
                        result['baseline_variance'] = f"{px_dev:.1f}"
                        break
                break

    def select_current_session(self):
        """Switch back to current session view."""
        self.batch_combo.setCurrentIndex(0)
