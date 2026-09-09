"""
Results List - Tree/table view of processed bills.
"""

import sys
import csv
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLabel, QLineEdit, QComboBox, QPushButton, QMenu, QHeaderView,
    QInputDialog, QDialog, QCheckBox, QDialogButtonBox
)
from PySide6.QtCore import Qt, Signal, Slot, QSettings, QEvent, QByteArray

# Bump when the set/order of columns changes, to invalidate an incompatible
# saved header layout instead of restoring it onto the wrong columns.
_HEADER_STATE_VERSION = 1
from PySide6.QtGui import QColor, QBrush, QAction, QIcon

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pattern_engine_v3 import PatternEngineV3 as PatternEngine

from settings_manager import get_settings
from gui.correction_dialog import CorrectionDialog, ReviewNoteDialog
from debug_logger import dlog, fingerprint


def _natural_sort_key(text: str):
    """Split a string into text/number runs so it sorts the way people expect.

    "A 9" sorts before "A 10" (the number run compares numerically), while the
    letter prefix still groups alphabetically. Each token is a (kind, num, text)
    tuple so int and str parts never get compared directly.
    """
    key = []
    for tok in re.split(r'(\d+)', text or ''):
        if tok.isdigit():
            key.append((1, int(tok), ''))       # numeric run
        elif tok:
            key.append((0, 0, tok.upper()))      # text run (case-insensitive)
    return key


class NumericTreeWidgetItem(QTreeWidgetItem):
    """TreeWidgetItem that sorts numerically for specific columns."""

    # Columns that should be sorted numerically (by index)
    # Back Plate (11) is digits-only, so it must sort numerically or "10" lands
    # before "9".
    NUMERIC_COLUMNS = {0, 4, 5, 6, 7, 11}  # Position, GPT, Shift X%, Shift Y%, Seal %, Back Plate

    # Columns that should use natural sort (letters group, embedded numbers sort
    # numerically). Front Plate (10) is alphanumeric like "A 9" / "A 10".
    NATURAL_COLUMNS = {10}  # Front Plate

    def __lt__(self, other):
        column = self.treeWidget().sortColumn() if self.treeWidget() else 0
        if column in self.NUMERIC_COLUMNS:
            try:
                # Parse the leading number, tolerating a trailing label like the
                # GPT column's "4.0 strong" and signed values like "+5.2" / "-3.1".
                def _lead(t):
                    m = re.match(r'\s*([-+]?[\d.]+)', t)
                    return float(m.group(1).replace('+', '')) if m else float('-inf')
                return _lead(self.text(column)) < _lead(other.text(column))
            except ValueError:
                pass
        elif column in self.NATURAL_COLUMNS:
            return _natural_sort_key(self.text(column)) < _natural_sort_key(other.text(column))
        # Fall back to string comparison (avoid super().__lt__ which can recurse)
        return self.text(column) < other.text(column)


# Gas-pump severity: the "strong" tier boundary as a multiple of the detect
# threshold (the slider value). Keep in sync with patterns/core/gas_pump.lua.
GAS_PUMP_STRONG_RATIO = 1.3


def _gpt_cell_text(px_dev: float, threshold: float) -> str:
    """GPT column text with a severity tier appended once past the detect
    threshold: below threshold = plain number (not a gas pump); at/above =
    "<px> mild", and at/above threshold*ratio = "<px> strong". The number stays
    first so the column still sorts numerically (see NumericTreeWidgetItem)."""
    base = f"{px_dev:.1f}"
    if threshold and px_dev >= threshold:
        return f"{base} strong" if px_dev >= threshold * GAS_PUMP_STRONG_RATIO else f"{base} mild"
    return base


class SetPatternsDialog(QDialog):
    """Checkbox dialog to pick one or more patterns for a bill.

    Each selected pattern produces its own serial overlay crop, and all selected
    names go on the label. A checkbox dialog (rather than the old menu) lets the
    user tick several at once without the menu closing after each pick.
    """

    def __init__(self, patterns, selected, parent=None, labels=None):
        super().__init__(parent)
        self.setWindowTitle("Set Pattern(s)")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select pattern(s) — one crop per pattern, all on the label:"))

        # Show friendly display names on the checkboxes, but store/return the
        # internal pattern IDs (so labels & crops resolve correctly).
        labels = labels or {}
        self._checks = []
        for p in patterns:
            cb = QCheckBox(labels.get(p, p))
            cb.setChecked(p in selected)
            layout.addWidget(cb)
            self._checks.append((p, cb))

        if not patterns:
            layout.addWidget(QLabel("(No detected patterns — add a custom one below.)"))

        layout.addWidget(QLabel("Custom (comma-separated):"))
        self._custom = QLineEdit()
        extra = [p for p in selected if p not in patterns]
        if extra:
            self._custom.setText(", ".join(extra))
        layout.addWidget(self._custom)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_patterns(self) -> list:
        """Checked patterns followed by any custom entries, de-duplicated."""
        out = [p for p, cb in self._checks if cb.isChecked()]
        for c in (t.strip() for t in self._custom.text().split(",")):
            if c and c not in out:
                out.append(c)
        return out


class ResultsList(QWidget):
    """List of processing results with filtering and sorting."""

    # Signals
    item_selected = Signal(dict)  # Emits the selected result
    correction_applied = Signal(str, str, str)  # filename, original, corrected
    batch_changed = Signal(str)  # Emits batch path when changed (empty for current session)
    crop_requested = Signal(list)  # Emits list of results to crop
    status_changed = Signal()  # Emits when review status fields change (viewed, cropped, etc.)
    serial_lookup_requested = Signal()  # Emits when the Serial Lookup button is clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results: List[dict] = []
        self.filtered_results: List[dict] = []
        self.filters: Dict[str, bool] = {}
        self.pattern_engine = PatternEngine()
        self.settings = get_settings()
        # View-set (Patterns-column A/B) state.
        self._view_set = None          # None => live enabled set (no override)
        self._view_cache = {}          # set label -> {result_key: matches str}
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
        # Click-only focus so arrow-key navigation in the results list can't
        # bleed into the dropdown and silently load an archive (a focused combo
        # box changes its selection on Up/Down without opening the popup).
        self.batch_combo.setFocusPolicy(Qt.ClickFocus)
        # Log when the dropdown actually gains focus, to confirm the fix holds.
        self.batch_combo.installEventFilter(self)
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

        # View-set dropdown: re-show the Patterns column against a chosen pattern
        # set (Essentials vs the original enabled set, saved presets, or Full)
        # WITHOUT re-processing — a fast A/B "flicker" compare of the same bills.
        self.view_set_combo = QComboBox()
        self.view_set_combo.setMinimumWidth(150)
        self.view_set_combo.setToolTip(
            "Show the Patterns column as a chosen pattern set would match it. "
            "Bills and order stay put — only the Patterns column changes — so "
            "you can flip between sets to compare. Does not change your enabled "
            "patterns or processing.")
        self.view_set_combo.currentIndexChanged.connect(self._on_view_set_changed)
        filter_layout.addWidget(self.view_set_combo)
        self._refresh_view_set_combo()

        # Lock order: freeze the current row order so switching view-sets (or
        # re-sorting) doesn't reshuffle rows — for steady side-by-side compare.
        self.lock_order_btn = QPushButton("🔒 Lock order")
        self.lock_order_btn.setCheckable(True)
        self.lock_order_btn.setToolTip(
            "Freeze the current row order so flipping between view-sets doesn't "
            "reshuffle the rows (even when sorted by Patterns). Toggle off to "
            "sort again. Temporary — for comparing sets side by side.")
        self.lock_order_btn.toggled.connect(self._toggle_lock_order)
        filter_layout.addWidget(self.lock_order_btn)

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

        # Serial Lookup button -- type any serial and see the patterns it matches
        self.serial_lookup_btn = QPushButton("Serial Lookup")
        self.serial_lookup_btn.setToolTip("Type a serial number and see which patterns it matches (Ctrl+L)")
        self.serial_lookup_btn.clicked.connect(lambda: self.serial_lookup_requested.emit())
        filter_layout.addWidget(self.serial_lookup_btn)

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
        self.tree.setHeaderLabels(["#", "Serial", "Patterns", "Conf", "GPT", "Shift X%", "Shift Y%", "Seal %", "Est. Price", "Series", "Front Plate", "Back Plate", "Mule? (exp)", "Mismatch?", "Status"])
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
            12: "EXPERIMENTAL hint, not a verdict: an old-era note (pre-1960s series) whose back-plate font reads small. True mules are a micro/macro plate-number font mismatch and vary by denomination -- verify with the plate magnifier (press M). Modern notes never flag.",
            13: "Mismatched serial numbers (two different serials detected on front)",
            14: "Status flags: ✓=queued, V=viewed, C=cropped, R=sent for review, ⟳=seen in a previous scan",
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
        header.setSectionsMovable(True)  # allow drag-reordering (and persist it)

        # Enable right-click context menu on header to hide columns
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(self._show_header_context_menu)

        # Callback for notifying main window when column visibility changes
        self._on_column_visibility_changed = None

        # Restore the saved column layout (order + widths + visibility), or apply
        # defaults. Done before connecting the save signals so restoring doesn't
        # re-trigger a save.
        self._restore_header_state()

        # Persist the layout whenever the user reorders or resizes columns.
        header.sectionResized.connect(self._save_header_state)
        header.sectionMoved.connect(self._save_header_state)

        layout.addWidget(self.tree)

        # Summary bar
        self.summary_label = QLabel("0 bills")
        layout.addWidget(self.summary_label)

    def _restore_header_state(self):
        """Restore the saved column layout (order + widths + visibility) via
        QHeaderView.restoreState, or apply defaults if there is no compatible
        saved layout."""
        settings = QSettings("DollarDetective", "ResultsList")
        blob = settings.value("header_state")
        saved_cols = settings.value("header_state_columns", 0, type=int)
        saved_ver = settings.value("header_state_version", 0, type=int)
        header = self.tree.header()

        if (blob is not None
                and saved_cols == self.tree.columnCount()
                and saved_ver == _HEADER_STATE_VERSION):
            try:
                state = blob if isinstance(blob, QByteArray) else QByteArray(blob)
                if header.restoreState(state):
                    return  # order + widths + visibility restored
            except Exception:
                pass  # corrupt/incompatible state -> fall through to defaults

        self._apply_default_columns()

    def _apply_default_columns(self):
        """Apply the default column widths and order (used when there is no saved
        layout)."""
        header = self.tree.header()
        # Default widths per logical column: #, Serial, Patterns, Conf, GPT,
        # Shift X%, Shift Y%, Seal %, Est. Price, Series, Front Plate, Back Plate,
        # Mule?, Mismatch?, Status
        defaults = [35, 130, 150, 50, 55, 50, 50, 45, 100, 60, 70, 60, 45, 55, 50]
        for i in range(self.tree.columnCount()):
            self.tree.setColumnWidth(i, defaults[i] if i < len(defaults) else 60)
        # Default order: Status (logical 14) sits just after the "#" column.
        header.moveSection(header.visualIndex(14), 1)

    def _save_header_state(self, *args):
        """Persist the full column layout (order + widths + visibility)."""
        settings = QSettings("DollarDetective", "ResultsList")
        settings.setValue("header_state", self.tree.header().saveState())
        settings.setValue("header_state_columns", self.tree.columnCount())
        settings.setValue("header_state_version", _HEADER_STATE_VERSION)

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
        self._save_header_state()

    def is_column_visible(self, column: int) -> bool:
        """Check if a column is visible."""
        return not self.tree.header().isSectionHidden(column)

    def eventFilter(self, obj, event):
        """Log when the batch dropdown gains focus (diagnostic for the
        accidental archive-load issue). Never consumes the event."""
        if obj is self.batch_combo and event.type() == QEvent.FocusIn:
            try:
                dlog("batch_combo.focus_in", reason=int(event.reason()))
            except Exception:
                pass
        return super().eventFilter(obj, event)

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
        dlog("results_list.set_results", was=fingerprint(self.results),
             now=fingerprint(results))
        self.results = results
        # New batch => cached view-sets are stale; refresh available presets.
        self._view_cache.clear()
        self._refresh_view_set_combo()
        self._rebuild_pattern_filter()
        self._apply_filters()

    def clear(self):
        """Clear all results."""
        dlog("results_list.clear", was=fingerprint(self.results))
        self.results = []
        self.filtered_results = []
        self._view_cache.clear()
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

            # Patterns - show display names in the column. Routed through the
            # view-set accessor so a chosen preset re-shows this column without
            # touching the stored (live enabled-set) classification.
            patterns = self._patterns_for_result(result)
            item.setText(2, self._format_patterns_display(patterns))

            # Confidence
            conf = result.get('confidence', '0.00')
            item.setText(3, str(conf))

            # Pixel Deviation (for gas pump detection)
            baseline_variance = result.get('baseline_variance', '0.0')
            try:
                px_dev = float(baseline_variance)
                try:
                    gp_threshold = self.pattern_engine.get_gas_pump_threshold()
                except Exception:
                    gp_threshold = 3.5
                item.setText(4, _gpt_cell_text(px_dev, gp_threshold))
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
                item.setText(12, "Check")  # experimental hint -> verify with magnifier (M)
            else:
                item.setText(12, "")

            # Mismatch detection column
            if result.get('serial_mismatch', False):
                item.setText(13, "Yes")
            else:
                item.setText(13, "")

            # Status column (review tracking)
            item.setText(14, self._build_status_text(result))

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
            # Seen-before detail on the Status column (from the bill ledger).
            if result.get('seen_before'):
                times = result.get('times_seen') or 2
                seen_msg = f"⟳ Seen before — scanned {times}× total"
                if result.get('prev_kept'):
                    seen_msg += " (you kept it previously)"
                item.setToolTip(14, seen_msg)

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

            # Remember the base Patterns-cell colors so the compare-diff
            # highlight can toggle amber on/off and restore them cleanly.
            item.setData(2, Qt.UserRole + 1, item.background(2))
            item.setData(2, Qt.UserRole + 2, item.foreground(2))

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
                    return
            # No authoritative row matched this edit -> it lives only on a
            # throwaway tree copy and will be lost on the next repopulate.
            dlog("sync.MISS", field=field, reason="no_match_in_results",
                 front_file=front_file, n_results=len(self.results))
        else:
            dlog("sync.MISS", field=field, reason="empty_front_file")

    @staticmethod
    def _build_status_text(result: dict) -> str:
        """Status-column flags: \u2713=queued, V=viewed, C=cropped, R=review,
        \u27f3=seen in a previous scan (from the bill ledger)."""
        parts = []
        if result.get('checked'):
            parts.append('\u2713')
        auto = ''
        if result.get('viewed'):
            auto += 'V'
        if result.get('cropped'):
            auto += 'C'
        if result.get('sent_for_review'):
            auto += 'R'
        if result.get('seen_before'):
            auto += '\u27f3'
        if auto:
            parts.append(auto)
        return ' '.join(parts)

    def _update_status_cell(self, item, result: dict):
        """Update the status column text for a single tree item.

        Note: PySide6's data()/setData() copies dicts, so the caller must
        pass the already-modified result AND store it back via setData().
        """
        item.setText(14, self._build_status_text(result))  # Column 14 = Status
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
                dlog("action.viewed", front_file=result.get('front_file'),
                     position=result.get('position'))
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
            # Single selection - open the multi-select "Set Pattern(s)..." dialog
            selected = self._current_pattern_overrides(result)
            label = "Set Pattern(s)..."
            if selected:
                label = f"Set Pattern(s)... ({len(selected)} selected)"
            set_patterns_action = QAction(label, self)
            set_patterns_action.triggered.connect(
                lambda: self._open_set_patterns_dialog(result)
            )
            menu.addAction(set_patterns_action)

            # "Set Note..." option
            note_action = QAction("Set Note...", self)
            note_action.triggered.connect(lambda: self._set_note(result))
            menu.addAction(note_action)

            # "Suggest Note" - auto-derive the line-3 feature note from the match
            suggest_action = QAction("Suggest Note", self)
            suggest_action.setToolTip("Auto-fill the note from the pattern (ladder digits, "
                                      "grouping, low-run size)")
            suggest_action.triggered.connect(lambda: self._suggest_note(result))
            menu.addAction(suggest_action)
        else:
            # Multi-select: fill empty notes on all selected bills at once.
            suggest_multi = QAction("Suggest Notes (fill empty)", self)
            suggest_multi.triggered.connect(lambda: self._suggest_notes_multi(selected_results))
            menu.addAction(suggest_multi)

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

    def _current_pattern_overrides(self, result: dict) -> list:
        """The bill's currently-selected override pattern(s), as a list.

        Reads the new multi-select ``pattern_overrides`` list, falling back to the
        legacy single ``pattern_override`` for bills saved before multi-select.
        """
        overrides = result.get('pattern_overrides')
        if overrides:
            return [p.strip() for p in overrides.split(',') if p.strip()]
        single = result.get('pattern_override')
        return [single] if single else []

    def _open_set_patterns_dialog(self, result: dict):
        """Open the checkbox dialog to choose the bill's pattern(s)."""
        patterns = [p.strip() for p in (result.get('fancy_types', '') or '').split(',') if p.strip()]
        selected = self._current_pattern_overrides(result)
        labels = {p: self._get_display_name(p) for p in patterns}
        dialog = SetPatternsDialog(patterns, selected, self, labels=labels)
        if dialog.exec():
            self._set_pattern_overrides(result, dialog.selected_patterns())

    def _set_pattern_overrides(self, result: dict, patterns: list):
        """Store the selected override pattern(s) on the result (and mirror the
        first onto the legacy single field for back-compat). Empty list clears it.
        Persisted via session recovery."""
        front_file = result.get('front_file')
        if not front_file:
            return
        patterns = [p for p in patterns if p]
        dlog("action.set_patterns", front_file=front_file,
             position=result.get('position'), value=", ".join(patterns))

        def apply(r):
            if patterns:
                # Comma-joined string (like fancy_types) so it round-trips through
                # the CSV session/archive persistence.
                r['pattern_overrides'] = ', '.join(patterns)
                r['pattern_override'] = patterns[0]  # legacy single-value mirror
                # Setting a pattern means you want this bill cropped, so queue it
                # (check it) automatically -- no need to also hit space. Clearing
                # the pattern leaves the checked state untouched.
                r['checked'] = True
            else:
                r.pop('pattern_overrides', None)
                r.pop('pattern_override', None)

        # Update the authoritative results list
        for r in self.results:
            if r.get('front_file') == front_file:
                apply(r)
                break

        # Update the tree item data
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item_result = item.data(0, Qt.UserRole)
            if item_result and item_result.get('front_file') == front_file:
                apply(item_result)
                item.setData(0, Qt.UserRole, item_result)
                if patterns:
                    # Reflect the auto-check in the row + persistence + summary.
                    self._sync_result_field(item_result, 'checked', True)
                    self._update_status_cell(item, item_result)
                break

        if patterns:
            self._update_summary()
        # Emit status_changed to trigger autosave
        self.status_changed.emit()

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
            dlog("action.note.DROPPED", reason="empty_front_file",
                 position=result.get('position'))
            return

        dlog("action.note", front_file=front_file,
             position=result.get('position'), value=note)

        self.apply_note(front_file, note)

    def apply_note(self, front_file: str, note: str):
        """Set (or clear) a bill's note by front_file and trigger autosave.

        Shared by the right-click "Set Note" dialog and the Label Preview tool so
        both write through the same path (authoritative list + tree item data +
        status_changed for autosave).
        """
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

    def _compute_suggestion(self, result: dict) -> str:
        """The auto-derived line-3 note for a bill (or '' if none applies)."""
        from .label_suggest import suggest_annotation
        serial = result.get('serial', '') or ''
        if not serial:
            return ''
        known = [p.strip() for p in (result.get('fancy_types', '') or '').split(',') if p.strip()]
        try:
            matches = self.pattern_engine.classify(serial)
        except Exception:
            matches = []
        return suggest_annotation(serial, matches, known_names=known)

    def _suggest_note(self, result: dict):
        """Right-click 'Suggest Note': fill this bill's note from its pattern."""
        from PySide6.QtWidgets import QMessageBox
        sug = self._compute_suggestion(result)
        if not sug:
            QMessageBox.information(self, "Suggest Note",
                                    "No note suggestion for this bill's pattern.")
            return
        existing = (result.get('note', '') or '').strip()
        if existing and QMessageBox.question(
                self, "Suggest Note",
                f"Replace the existing note with the suggestion?\n\nSuggested: {sug}",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.apply_note(result.get('front_file'), sug)

    def _suggest_notes_multi(self, results: list):
        """Right-click 'Suggest Notes': fill EMPTY notes on selected bills."""
        from PySide6.QtWidgets import QMessageBox
        filled = skipped = 0
        for r in results:
            if (r.get('note', '') or '').strip():
                skipped += 1
                continue
            sug = self._compute_suggestion(r)
            if not sug:
                continue
            self.apply_note(r.get('front_file'), sug)
            filled += 1
        msg = f"Filled {filled} note(s) from pattern data."
        if skipped:
            msg += f"\nSkipped {skipped} that already had a note."
        QMessageBox.information(self, "Suggest Notes", msg)

    def apply_label_lines(self, front_file: str, lines):
        """Set (or clear) a bill's per-bill label override and trigger autosave.

        `lines` is a list of label text lines (label-only edits from the Label
        Preview tool) or None to clear the override and fall back to auto text.
        Writes through the same path as notes (authoritative list + tree item +
        status_changed).
        """
        if not front_file:
            return
        for r in self.results:
            if r.get('front_file') == front_file:
                if lines:
                    r['label_lines'] = list(lines)
                elif 'label_lines' in r:
                    del r['label_lines']
                break
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item_result = item.data(0, Qt.UserRole)
            if item_result and item_result.get('front_file') == front_file:
                if lines:
                    item_result['label_lines'] = list(lines)
                elif 'label_lines' in item_result:
                    del item_result['label_lines']
                item.setData(0, Qt.UserRole, item_result)
                break
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

        # Review folder: the user-configured Review Directory if set, else the
        # writable per-user data dir (repo root in dev; ~/.config/... when frozen).
        from resource_path import user_data_dir
        configured = (self.settings.ui.review_directory or "").strip()
        review_folder = Path(configured) if configured else (user_data_dir() / "review")
        review_folder.mkdir(parents=True, exist_ok=True)

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
                dlog("action.toggle_checked", front_file=result.get('front_file'),
                     position=result.get('position'), new_val=new_val)
                result['checked'] = new_val
                self._sync_result_field(result, 'checked', new_val)
                self._update_status_cell(item, result)
        self._update_summary()

    def mark_cropped(self, results: list):
        """Mark given results as cropped and clear checked flag."""
        dlog("action.mark_cropped", count=len(results),
             front_files=[r.get('front_file') for r in results][:10])
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
        archive_dir = self.settings.processing.archive_directory
        if not archive_dir:
            # Fall back to the last-used input dir's archive folder
            last_input = self.settings.ui.last_input_dir
            archive_dir = str(Path(last_input) / "archive") if last_input else ""

        archive_path = Path(archive_dir) if archive_dir else None
        if archive_path and archive_path.exists():
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
        dlog("batch_combo.changed", index=index, batch_path=batch_path or "(current session)")

        if not batch_path:
            # Current session selected. NOTE: this does NOT restore the live
            # session's dicts into the list, so the display can diverge from
            # MainWindow.current_results after having viewed an archived batch.
            self._current_batch_path = None
            self.save_csv_btn.setEnabled(False)
            self.batch_changed.emit("")
        else:
            # Archived batch selected. This replaces self.results with fresh
            # CSV dicts -> object-sharing with current_results is broken; any
            # edits made while a batch is selected land only on these dicts.
            dlog("batch.load.START", batch=str(batch_path),
                 before=fingerprint(self.results))
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
                    'note', 'pattern_override', 'pattern_overrides'
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
                try:
                    gp_threshold = self.pattern_engine.get_gas_pump_threshold()
                except Exception:
                    gp_threshold = 3.5
                item.setText(4, _gpt_cell_text(px_dev, gp_threshold))
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

    # ------------------------------------------------------------------ #
    # View-set (Patterns-column A/B compare)
    # ------------------------------------------------------------------ #
    _VIEW_LIVE_LABEL = "View: Enabled set"
    _VIEW_FULL_LABEL = "View: ★ Full library"
    _VIEW_ESSENTIALS_LABEL = "View: ★ Essentials"

    def _essentials_view_states(self):
        """Bundled Essentials selection {pattern: enabled}, or None."""
        import json
        path = Path(__file__).resolve().parent.parent / "essentials_default.json"
        try:
            return json.loads(path.read_text(encoding="utf-8")).get(
                "pattern_states") or {}
        except Exception:
            return None

    def _refresh_view_set_combo(self):
        """Populate the view-set dropdown: live enabled set + built-ins +
        saved selection presets (mirrors Pattern Manager)."""
        combo = getattr(self, "view_set_combo", None)
        if combo is None:
            return
        prev = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(self._VIEW_LIVE_LABEL)
        combo.addItem(self._VIEW_FULL_LABEL)
        if self._essentials_view_states() is not None:
            combo.addItem(self._VIEW_ESSENTIALS_LABEL)
        try:
            for name in sorted(self.settings.get_selection_presets()):
                combo.addItem(name)
        except Exception:
            pass
        idx = combo.findText(prev)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def _resolve_view_allowed(self, label):
        """The set of enabled pattern names for a view-set label, or None for
        the live enabled set (no override)."""
        if not label or label == self._VIEW_LIVE_LABEL:
            return None
        if label == self._VIEW_FULL_LABEL:
            return set(self.pattern_engine.lua_patterns.keys())
        if label == self._VIEW_ESSENTIALS_LABEL:
            states = self._essentials_view_states() or {}
            return {n for n, on in states.items() if on}
        preset = self.settings.get_selection_presets().get(label)
        if not preset:
            return None
        states = preset.get("pattern_states", {})
        return {n for n, on in states.items() if on}

    @staticmethod
    def _result_key(result):
        return result.get('front_file') or \
            f"{result.get('position', '')}:{result.get('serial', '')}"

    def _patterns_for_result(self, result):
        """The Patterns-column string to show under the current view-set: the
        stored live classification, or a cached re-match against a preset."""
        if self._view_set is None:
            return result.get('fancy_types', '')
        cache = self._view_cache.get(self._view_set, {})
        return cache.get(self._result_key(result), '')

    def _compute_view_cache(self, label, allowed):
        """Re-match every result against `allowed` (view only — never mutates
        the stored classification), cached so re-toggling is instant."""
        cache = {}
        for result in self.results:
            serial = result.get('serial', '')
            if not serial:
                cache[self._result_key(result)] = ''
                continue
            metadata = {
                'baseline_variance': float(result.get('baseline_variance', 0) or 0),
                'gas_pump_threshold': self.pattern_engine.get_gas_pump_threshold(),
                'series_year': result.get('series_year', ''),
                'front_plate': result.get('front_plate', ''),
                'back_plate': result.get('back_plate', ''),
            }
            matches = self.pattern_engine.classify_reference(
                serial, allowed, metadata)
            cache[self._result_key(result)] = ', '.join(matches) if matches else ''
        self._view_cache[label] = cache

    def _on_view_set_changed(self):
        """Switch the Patterns column to the chosen view-set (compute+cache on
        first use, then just re-render — same bills, same order)."""
        from PySide6.QtWidgets import QApplication
        label = self.view_set_combo.currentText()
        if label == self._VIEW_LIVE_LABEL:
            self._view_set = None
        else:
            allowed = self._resolve_view_allowed(label)
            if allowed is None:
                self._view_set = None
            else:
                self._view_set = label
                if label not in self._view_cache:
                    QApplication.setOverrideCursor(Qt.WaitCursor)
                    try:
                        self._compute_view_cache(label, allowed)
                    finally:
                        QApplication.restoreOverrideCursor()
        # Re-render only the Patterns column in place (keep order + selection).
        # Bills whose match set DIFFERS from the live enabled set get an amber
        # cell so a flick between sets shows exactly what changed; matching
        # bills restore the base cell color captured at render time.
        amber, black = QBrush(QColor(255, 193, 7)), QBrush(QColor(0, 0, 0))
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            result = item.data(0, Qt.UserRole)
            if result is None:
                continue
            item.setText(2, self._format_patterns_display(
                self._patterns_for_result(result)) or "-")
            if self._view_differs(result):
                item.setBackground(2, amber)
                item.setForeground(2, black)
            else:
                base_bg = item.data(2, Qt.UserRole + 1)
                base_fg = item.data(2, Qt.UserRole + 2)
                item.setBackground(2, base_bg if base_bg is not None else QBrush())
                item.setForeground(2, base_fg if base_fg is not None else QBrush())
        self._update_summary()

    def _view_differs(self, result):
        """True if the viewed set matches this bill differently than the live
        enabled set (only meaningful while viewing a non-live set)."""
        if self._view_set is None:
            return False

        def names(s):
            return {n.strip() for n in (s or '').split(',') if n.strip()}
        live = names(result.get('fancy_types', ''))
        shown = names(self._view_cache.get(self._view_set, {}).get(
            self._result_key(result), ''))
        return live != shown

    def _toggle_lock_order(self, checked):
        """Freeze/unfreeze row order for steady side-by-side compare, and mark
        the locked (last-sorted) column header with a lock so the state is
        obvious even though the sort arrow disappears."""
        header = self.tree.header()
        hitem = self.tree.headerItem()
        if checked:
            col = header.sortIndicatorSection() if header else -1
            self.tree.setSortingEnabled(False)
            self.lock_order_btn.setText("🔒 Order locked")
            if hitem is not None and col is not None and col >= 0:
                self._locked_col = col
                self._locked_col_text = hitem.text(col)
                hitem.setText(col, f"🔒 {self._locked_col_text}")
        else:
            col = getattr(self, "_locked_col", None)
            if hitem is not None and col is not None and col >= 0:
                hitem.setText(col, getattr(self, "_locked_col_text", hitem.text(col)))
            self._locked_col = None
            self.tree.setSortingEnabled(True)
            self.lock_order_btn.setText("🔒 Lock order")

    def _reclassify_all(self):
        """Re-run pattern matching on all results."""
        # A re-classify changes the live set, so any cached view-sets are stale.
        self._view_cache.clear()
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
