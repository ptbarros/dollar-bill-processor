"""
Pattern Dialog - Manage pattern enable/disable and testing.

Supports both YAML-based simple rules and Lua script patterns.
"""

import sys
import re
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QGroupBox, QLineEdit, QPushButton, QDialogButtonBox, QLabel,
    QTextEdit, QSplitter, QHeaderView, QCheckBox, QListWidget,
    QListWidgetItem, QFormLayout, QComboBox, QMessageBox, QColorDialog,
    QTabWidget, QWidget, QSpinBox, QFrame, QApplication, QPlainTextEdit,
    QInputDialog, QMenu, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat, QFontMetrics

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pattern_engine_v3 import PatternEngineV3 as PatternEngine
HAS_V3_ENGINE = True

from settings_manager import get_settings

# Import pattern recipes for wizard
from gui.pattern_recipes import get_all_recipes, ParameterDef

# Import AI pattern generator
from gui.ai_pattern_generator import AIPatternGenerator

# Try to import QScintilla for better code editing
try:
    from PyQt5.Qsci import QsciScintilla, QsciLexerLua
    HAS_QSCINTILLA = True
except ImportError:
    HAS_QSCINTILLA = False


class ColorPickerDialog(QDialog):
    """Color picker dialog with Clear Color option."""

    # Result codes
    CANCELLED = 0
    COLOR_SELECTED = 1
    COLOR_CLEARED = 2

    def __init__(self, parent=None, title="Choose Color", initial_color=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.selected_color = None
        self.result_code = self.CANCELLED

        layout = QVBoxLayout(self)

        # Color dialog widget (non-modal, embedded)
        self.color_dialog = QColorDialog(self)
        self.color_dialog.setWindowFlags(Qt.Widget)  # Embed as widget, not separate window
        self.color_dialog.setOptions(
            QColorDialog.DontUseNativeDialog |
            QColorDialog.NoButtons  # We'll add our own buttons
        )
        if initial_color:
            self.color_dialog.setCurrentColor(initial_color)
        layout.addWidget(self.color_dialog)

        # Button row
        btn_layout = QHBoxLayout()

        clear_btn = QPushButton("Clear Color")
        clear_btn.clicked.connect(self._on_clear)
        btn_layout.addWidget(clear_btn)

        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_ok)
        btn_layout.addWidget(ok_btn)

        layout.addLayout(btn_layout)

    def _on_ok(self):
        self.selected_color = self.color_dialog.currentColor()
        self.result_code = self.COLOR_SELECTED
        self.accept()

    def _on_clear(self):
        self.selected_color = None
        self.result_code = self.COLOR_CLEARED
        self.accept()

    @staticmethod
    def getColor(initial=None, parent=None, title="Choose Color"):
        """Show dialog and return (result_code, color).

        Returns:
            (COLOR_SELECTED, QColor) - User picked a color
            (COLOR_CLEARED, None) - User clicked Clear Color
            (CANCELLED, None) - User cancelled
        """
        dialog = ColorPickerDialog(parent, title, initial)
        dialog.exec()
        return dialog.result_code, dialog.selected_color


class PatternDialog(QDialog):
    """Dialog for managing patterns and testing serials."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = PatternEngine()
        self.settings = get_settings()
        self._patterns_modified = False  # Track if patterns were changed

        self.setWindowTitle("Pattern Manager")
        self.setMinimumSize(900, 600)
        self._setup_ui()
        self._load_patterns()
        self._restore_window_geometry()

    def patterns_were_modified(self) -> bool:
        """Return True if patterns were created, deleted, or modified during this session."""
        return self._patterns_modified

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Pattern list
        left_panel = QGroupBox("Patterns")
        left_layout = QVBoxLayout(left_panel)

        # Search/filter
        filter_layout = QHBoxLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter patterns...")
        self.filter_edit.textChanged.connect(self._filter_patterns)
        filter_layout.addWidget(self.filter_edit)

        self.show_disabled_check = QCheckBox("Show disabled")
        self.show_disabled_check.setChecked(True)
        self.show_disabled_check.stateChanged.connect(self._filter_patterns)
        filter_layout.addWidget(self.show_disabled_check)

        left_layout.addLayout(filter_layout)

        # Pattern tree - grouped by library, sortable by tier
        self.pattern_tree = QTreeWidget()
        self.pattern_tree.setHeaderLabels(["Pattern", "Tier", "Enabled", "Color", "Catalog"])
        self.pattern_tree.setRootIsDecorated(True)
        self.pattern_tree.itemChanged.connect(self._on_item_changed)
        self.pattern_tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.pattern_tree.itemDoubleClicked.connect(self._on_item_double_click)
        self.pattern_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.pattern_tree.customContextMenuRequested.connect(self._on_context_menu)
        self.pattern_tree.setSortingEnabled(True)

        header = self.pattern_tree.header()
        header.setSectionsMovable(True)  # Allow reordering columns by dragging
        header.setStretchLastSection(True)  # Last column fills remaining space
        header.setSortIndicatorShown(True)
        # All columns interactive (draggable) except last which stretches
        header.setSectionResizeMode(0, QHeaderView.Interactive)  # Pattern
        header.setSectionResizeMode(1, QHeaderView.Interactive)  # Tier
        header.setSectionResizeMode(2, QHeaderView.Interactive)  # Enabled
        header.setSectionResizeMode(3, QHeaderView.Interactive)  # Color
        header.setSectionResizeMode(4, QHeaderView.Stretch)      # Catalog

        # Restore saved column widths/order or use defaults
        self._restore_column_widths()

        # Save column widths and order when changed
        header.sectionResized.connect(self._save_column_widths)
        header.sectionMoved.connect(self._save_column_widths)

        left_layout.addWidget(self.pattern_tree)

        # Enable/disable all buttons
        btn_layout = QHBoxLayout()
        enable_all_btn = QPushButton("Enable All")
        enable_all_btn.clicked.connect(self._enable_all)
        btn_layout.addWidget(enable_all_btn)

        disable_all_btn = QPushButton("Disable All")
        disable_all_btn.clicked.connect(self._disable_all)
        btn_layout.addWidget(disable_all_btn)

        left_layout.addLayout(btn_layout)

        splitter.addWidget(left_panel)

        # Right panel - Details and testing
        right_panel = QGroupBox("Details & Testing")
        right_layout = QVBoxLayout(right_panel)

        # Pattern details
        details_group = QGroupBox("Selected Pattern")
        details_layout = QVBoxLayout(details_group)

        self.pattern_name_label = QLabel("-")
        self.pattern_name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        details_layout.addWidget(self.pattern_name_label)

        self.pattern_desc_label = QLabel("-")
        self.pattern_desc_label.setWordWrap(True)
        details_layout.addWidget(self.pattern_desc_label)

        self.pattern_tier_label = QLabel("Tier: -")
        details_layout.addWidget(self.pattern_tier_label)

        self.pattern_examples_label = QLabel("Examples: -")
        self.pattern_examples_label.setWordWrap(True)
        details_layout.addWidget(self.pattern_examples_label)

        self.pattern_odds_label = QLabel("Odds: -")
        self.pattern_odds_label.setStyleSheet("color: #1976D2; font-weight: bold;")
        details_layout.addWidget(self.pattern_odds_label)

        self.pattern_price_label = QLabel("Price: -")
        self.pattern_price_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
        details_layout.addWidget(self.pattern_price_label)

        # Threshold editor for height_ratio patterns (like GAS_PUMP)
        self.threshold_layout = QHBoxLayout()
        self.threshold_label = QLabel("Threshold:")
        self.threshold_layout.addWidget(self.threshold_label)
        self.threshold_edit = QLineEdit()
        self.threshold_edit.setPlaceholderText("e.g., 0.085")
        self.threshold_edit.setMaximumWidth(100)
        self.threshold_layout.addWidget(self.threshold_edit)
        self.threshold_save_btn = QPushButton("Save")
        self.threshold_save_btn.clicked.connect(self._save_threshold)
        self.threshold_layout.addWidget(self.threshold_save_btn)
        self.threshold_layout.addStretch()
        details_layout.addLayout(self.threshold_layout)

        # Initially hidden - shown only for height_ratio patterns
        self.threshold_label.hide()
        self.threshold_edit.hide()
        self.threshold_save_btn.hide()
        self._current_threshold_pattern = None

        # Lua script viewer - shown for patterns with Lua implementations
        self.lua_script_layout = QHBoxLayout()
        self.lua_script_label = QLabel("[Lua]")
        self.lua_script_label.setStyleSheet("color: #9C27B0; font-weight: bold;")
        self.lua_script_layout.addWidget(self.lua_script_label)
        self.view_script_btn = QPushButton("View Script")
        self.view_script_btn.setToolTip("View the Lua source code (can be copied as a template)")
        self.view_script_btn.clicked.connect(self._view_lua_script)
        self.lua_script_layout.addWidget(self.view_script_btn)

        self.delete_pattern_btn = QPushButton("Delete")
        self.delete_pattern_btn.setToolTip("Delete this user pattern")
        self.delete_pattern_btn.setStyleSheet("QPushButton { color: #d32f2f; }")
        self.delete_pattern_btn.clicked.connect(self._delete_user_pattern)
        self.lua_script_layout.addWidget(self.delete_pattern_btn)

        self.generate_serial_btn = QPushButton("Generate Random")
        self.generate_serial_btn.setToolTip("Generate a random matching serial")
        self.generate_serial_btn.clicked.connect(self._generate_test_serial)
        self.generate_serial_btn.setEnabled(False)
        self.lua_script_layout.addWidget(self.generate_serial_btn)

        # Test serial input
        self.test_serial_edit = QLineEdit()
        self.test_serial_edit.setPlaceholderText("Test serial (e.g., 12345678)")
        self.test_serial_edit.setMaximumWidth(180)
        self.test_serial_edit.textChanged.connect(self._on_test_serial_changed)
        self.lua_script_layout.addWidget(self.test_serial_edit)

        self.lua_script_layout.addStretch()
        details_layout.addLayout(self.lua_script_layout)

        # Initially hidden
        self.lua_script_label.hide()
        self.view_script_btn.hide()
        self.delete_pattern_btn.hide()
        self.generate_serial_btn.hide()
        self.test_serial_edit.hide()
        self._current_lua_pattern = None
        self._current_lua_editable = False  # True if pattern is not from 'core'

        # Pattern preview section
        self.pattern_preview = DigitPreviewWidget()
        self.pattern_preview.setMinimumHeight(100)
        details_layout.addWidget(self.pattern_preview)

        self.match_message_label = QLabel("")
        self.match_message_label.setWordWrap(True)
        self.match_message_label.setStyleSheet("color: #666; font-style: italic;")
        details_layout.addWidget(self.match_message_label)

        right_layout.addWidget(details_group)

        # Actions section
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)

        # Row 1: Library and Pattern creation
        row1 = QHBoxLayout()

        new_library_btn = QPushButton("New Library...")
        new_library_btn.setToolTip("Create a new pattern library folder")
        new_library_btn.clicked.connect(self._create_new_library)
        row1.addWidget(new_library_btn)

        new_pattern_btn = QPushButton("New Pattern...")
        new_pattern_btn.setToolTip("Create a new Lua pattern script")
        new_pattern_btn.clicked.connect(self._add_custom_pattern)
        row1.addWidget(new_pattern_btn)

        actions_layout.addLayout(row1)

        # Row 2: Documentation and utilities
        row2 = QHBoxLayout()

        api_docs_btn = QPushButton("API Docs")
        api_docs_btn.setToolTip("View Lua pattern scripting documentation")
        api_docs_btn.clicked.connect(self._show_api_docs)
        row2.addWidget(api_docs_btn)

        open_folder_btn = QPushButton("Open Patterns Folder")
        open_folder_btn.setToolTip("Open the patterns directory in file explorer")
        open_folder_btn.clicked.connect(self._open_patterns_folder)
        row2.addWidget(open_folder_btn)

        actions_layout.addLayout(row2)

        right_layout.addWidget(actions_group)

        right_layout.addStretch()

        splitter.addWidget(right_panel)

        # Set splitter sizes
        splitter.setSizes([400, 400])

        layout.addWidget(splitter)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._save_and_close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _restore_column_widths(self):
        """Restore saved column widths and order from settings."""
        header = self.pattern_tree.header()

        # Restore column order
        order = self.settings.get_custom_value('pattern_manager_column_order', None)
        if order and len(order) == 5:
            for visual_idx, logical_idx in enumerate(order):
                current_visual = header.visualIndex(logical_idx)
                if current_visual != visual_idx:
                    header.moveSection(current_visual, visual_idx)

        # Restore column widths
        widths = self.settings.get_custom_value('pattern_manager_columns_v3', None)
        if widths and len(widths) >= 4:
            self.pattern_tree.setColumnWidth(0, widths[0])  # Pattern
            self.pattern_tree.setColumnWidth(1, widths[1])  # Tier
            self.pattern_tree.setColumnWidth(2, widths[2])  # Enabled
            self.pattern_tree.setColumnWidth(3, widths[3])  # Color
            # Column 4 (Catalog) stretches to fill
        else:
            # Default widths
            self.pattern_tree.setColumnWidth(0, 200)  # Pattern
            self.pattern_tree.setColumnWidth(1, 40)   # Tier
            self.pattern_tree.setColumnWidth(2, 55)   # Enabled
            self.pattern_tree.setColumnWidth(3, 45)   # Color

    def _save_column_widths(self):
        """Save column widths and order to settings."""
        header = self.pattern_tree.header()

        # Save widths
        widths = [
            self.pattern_tree.columnWidth(0),
            self.pattern_tree.columnWidth(1),
            self.pattern_tree.columnWidth(2),
            self.pattern_tree.columnWidth(3),
        ]
        self.settings.set_custom_value('pattern_manager_columns_v3', widths)

        # Save column order (logical indices in visual order)
        order = [header.logicalIndex(i) for i in range(header.count())]
        self.settings.set_custom_value('pattern_manager_column_order', order)

    def _restore_window_geometry(self):
        """Restore saved window size and position."""
        geometry = self.settings.get_custom_value('pattern_manager_geometry', None)
        if geometry and len(geometry) == 4:
            x, y, w, h = geometry
            self.setGeometry(x, y, w, h)
        else:
            # Default size - wide enough to show all columns
            self.resize(1100, 700)

    def _save_window_geometry(self):
        """Save window size and position to settings."""
        geo = self.geometry()
        self.settings.set_custom_value('pattern_manager_geometry', [geo.x(), geo.y(), geo.width(), geo.height()])

    def _make_friendly_name(self, name: str) -> str:
        """Convert pattern name to friendly display name.

        Examples:
            LOW_RUN_6M -> Low Run 6M
            RADAR -> Radar
            DOUBLE_YEAR -> Double Year
        """
        words = name.replace('_', ' ').split()
        friendly_words = []
        for word in words:
            # Keep short alphanumeric tokens (like "6M", "3D") uppercase
            if len(word) <= 3 and any(c.isdigit() for c in word):
                friendly_words.append(word.upper())
            else:
                friendly_words.append(word.capitalize())
        return ' '.join(friendly_words)

    def _load_patterns(self):
        """Load patterns into the tree, grouped by library."""
        self.pattern_tree.clear()

        # Temporarily disable sorting while loading
        self.pattern_tree.setSortingEnabled(False)

        # Get Lua pattern info (includes file paths for library detection)
        lua_pattern_info = self.engine.lua_patterns

        # Group patterns by library - Lua patterns are the only source
        libraries = {}  # library_name -> [(name, defn, lua_info), ...]

        # Add all Lua patterns
        for name, lua_info in lua_pattern_info.items():

            # Extract library from file path
            library = 'user'  # Default for Lua patterns without recognized path
            if lua_info.file_path:
                path_parts = Path(lua_info.file_path).parts
                if 'patterns' in path_parts:
                    patterns_idx = path_parts.index('patterns')
                    if patterns_idx + 1 < len(path_parts):
                        library = path_parts[patterns_idx + 1]

            # Build defn from Lua info
            defn = {
                'description': lua_info.description,
                'tier': lua_info.tier,
                'examples': lua_info.examples,
                'odds': lua_info.odds,
                'price_range': lua_info.price,
            }

            if library not in libraries:
                libraries[library] = []
            libraries[library].append((name, defn, lua_info))

        # Library colors for visual distinction
        lib_colors = {
            'core': QColor("#607D8B"),   # Blue-gray
            'user': QColor("#4CAF50"),   # Green
        }

        # Sort libraries: core first, then alphabetically
        sorted_libs = sorted(libraries.keys(), key=lambda x: (x != 'core', x))

        for library in sorted_libs:
            patterns = libraries[library]
            lib_enabled = self.settings.get_library_enabled(library, default=True)

            # Create library header item (checkable)
            lib_item = QTreeWidgetItem()
            lib_item.setText(0, f"{library} ({len(patterns)} patterns)")
            lib_item.setData(0, Qt.UserRole, {'is_library': True, 'library': library})
            lib_item.setFlags(lib_item.flags() | Qt.ItemIsUserCheckable)
            lib_item.setCheckState(0, Qt.Checked if lib_enabled else Qt.Unchecked)

            # Style the library header
            font = lib_item.font(0)
            font.setBold(True)
            lib_item.setFont(0, font)
            if library in lib_colors:
                lib_item.setForeground(0, lib_colors[library])

            # Library color indicator (double-click to change)
            lib_color = self.settings.get_library_color(library)
            if lib_color:
                lib_item.setText(3, "●")
                lib_item.setForeground(3, QColor(lib_color))
            else:
                lib_item.setText(3, "○")
                lib_item.setForeground(3, QColor("#888888"))

            # Add patterns under this library
            for name, defn, lua_info in sorted(patterns, key=lambda x: x[0]):
                pattern_item = QTreeWidgetItem(lib_item)

                has_lua = lua_info is not None
                tier = defn.get('tier', 10)

                # Use display_name from Lua info if available, otherwise auto-generate
                if has_lua and lua_info.display_name:
                    friendly_name = lua_info.display_name
                else:
                    # Auto-generate: LOW_RUN_6M -> Low Run 6M
                    friendly_name = self._make_friendly_name(name)

                if has_lua:
                    shown_name = f"{friendly_name} [Lua]"
                else:
                    shown_name = friendly_name
                pattern_item.setText(0, shown_name)

                # Tier column (for sorting)
                pattern_item.setText(1, str(tier))
                pattern_item.setData(1, Qt.UserRole, tier)  # Store as int for proper sorting

                # Checkbox for enabled
                enabled = lua_info.enabled
                pattern_item.setCheckState(2, Qt.Checked if enabled else Qt.Unchecked)
                pattern_item.setData(0, Qt.UserRole, {'name': name, 'defn': defn, 'has_lua': has_lua, 'library': library})

                # Color indicator (double-click to change)
                color = self.settings.get_pattern_color(name)
                if color:
                    pattern_item.setText(3, "●")
                    pattern_item.setForeground(3, QColor(color))
                else:
                    pattern_item.setText(3, "○")
                    pattern_item.setForeground(3, QColor("#888888"))

                # Catalog location (double-click to edit)
                catalog = self.settings.get_pattern_catalog(name)
                pattern_item.setText(4, catalog if catalog else "-")

                # Purple tint for Lua patterns
                if has_lua:
                    pattern_item.setForeground(0, QColor("#9C27B0"))

            self.pattern_tree.addTopLevelItem(lib_item)

        self.pattern_tree.expandAll()
        self.pattern_tree.setSortingEnabled(True)

    def _filter_patterns(self):
        """Filter patterns based on search text."""
        filter_text = self.filter_edit.text().lower()
        show_disabled = self.show_disabled_check.isChecked()

        for i in range(self.pattern_tree.topLevelItemCount()):
            lib_item = self.pattern_tree.topLevelItem(i)
            lib_data = lib_item.data(0, Qt.UserRole)
            library = lib_data.get('library', '').lower() if lib_data else ''
            visible_children = 0

            for j in range(lib_item.childCount()):
                pattern_item = lib_item.child(j)
                data = pattern_item.data(0, Qt.UserRole)
                name = data['name'].lower()
                desc = data['defn'].get('description', '').lower()
                enabled = pattern_item.checkState(2) == Qt.Checked  # Column 2 = Enabled

                # Filter by text (searches pattern name, description, and library name)
                text_match = not filter_text or filter_text in name or filter_text in desc or filter_text in library

                # Filter by enabled state
                enabled_match = show_disabled or enabled

                visible = text_match and enabled_match
                pattern_item.setHidden(not visible)

                if visible:
                    visible_children += 1

            lib_item.setHidden(visible_children == 0)

    def _on_item_changed(self, item, column):
        """Handle item check state change."""
        data = item.data(0, Qt.UserRole)
        if not data:
            return

        # Handle library checkbox (column 0 for library headers)
        if data.get('is_library') and column == 0:
            lib_name = data['library']
            enabled = item.checkState(0) == Qt.Checked
            self.settings.set_library_enabled(lib_name, enabled)
            # Update all patterns in this library
            self._update_library_patterns(item, enabled)
            return

        # Handle pattern checkbox (column 2)
        if column != 2:
            return

        if data.get('is_library'):
            return

        name = data['name']
        enabled = item.checkState(2) == Qt.Checked
        self.engine.set_pattern_enabled(name, enabled)

    def _update_library_patterns(self, lib_item, enabled: bool):
        """Enable or disable all patterns under a library item.

        Clears individual pattern overrides so patterns inherit the library state.
        This ensures library checkbox state is respected on reload.
        """
        self.pattern_tree.blockSignals(True)
        for i in range(lib_item.childCount()):
            pattern_item = lib_item.child(i)
            pattern_item.setCheckState(2, Qt.Checked if enabled else Qt.Unchecked)
            data = pattern_item.data(0, Qt.UserRole)
            if data and 'name' in data:
                # Clear individual pattern state so it inherits from library
                # (only for v3 engine with clear_pattern_enabled method)
                if hasattr(self.engine, 'clear_pattern_enabled'):
                    self.engine.clear_pattern_enabled(data['name'])
                else:
                    self.engine.set_pattern_enabled(data['name'], enabled)
        self.pattern_tree.blockSignals(False)

    def _select_pattern_by_name(self, pattern_name: str):
        """Find and select a pattern in the tree by name."""
        if not pattern_name:
            return

        # Iterate through all library items
        for i in range(self.pattern_tree.topLevelItemCount()):
            lib_item = self.pattern_tree.topLevelItem(i)
            # Iterate through patterns in this library
            for j in range(lib_item.childCount()):
                pattern_item = lib_item.child(j)
                data = pattern_item.data(0, Qt.UserRole)
                if data and data.get('name') == pattern_name:
                    # Found it - expand parent and select
                    lib_item.setExpanded(True)
                    self.pattern_tree.setCurrentItem(pattern_item)
                    return

    def _on_item_double_click(self, item, column):
        """Handle double-click on color or catalog column."""
        data = item.data(0, Qt.UserRole)
        if not data:
            return

        # Handle library row
        if data.get('is_library'):
            if column == 3:
                # Library color column - show color picker
                lib_name = data['library']
                current_color = self.settings.get_library_color(lib_name)
                initial = QColor(current_color) if current_color else QColor("#2e7d32")
                result, color = ColorPickerDialog.getColor(initial, self, f"Choose color for {lib_name} library")

                if result == ColorPickerDialog.COLOR_SELECTED and color.isValid():
                    hex_color = color.name()
                    self.settings.set_library_color(lib_name, hex_color)
                    item.setText(3, "●")
                    item.setForeground(3, color)
                elif result == ColorPickerDialog.COLOR_CLEARED:
                    self._clear_library_color(item, lib_name)
            return

        name = data['name']

        if column == 3:
            # Pattern color column - show color picker
            current_color = self.settings.get_pattern_color(name)
            initial = QColor(current_color) if current_color else QColor("#2e7d32")
            result, color = ColorPickerDialog.getColor(initial, self, f"Choose color for {name}")

            if result == ColorPickerDialog.COLOR_SELECTED and color.isValid():
                hex_color = color.name()
                self.settings.set_pattern_color(name, hex_color)
                item.setText(3, "●")
                item.setForeground(3, color)
            elif result == ColorPickerDialog.COLOR_CLEARED:
                self._clear_pattern_color(item, name)

        elif column == 4:
            # Catalog column - show input dialog
            current_catalog = self.settings.get_pattern_catalog(name)
            catalog, ok = QInputDialog.getText(
                self, f"Catalog for {name}",
                "Enter catalog location (e.g., A1, B2, 12):",
                text=current_catalog
            )
            if ok:
                self.settings.set_pattern_catalog(name, catalog.strip())
                item.setText(4, catalog.strip() if catalog.strip() else "-")

    def _on_context_menu(self, pos):
        """Show context menu for clearing colors."""
        item = self.pattern_tree.itemAt(pos)
        if not item:
            return

        data = item.data(0, Qt.UserRole)
        if not data:
            return

        menu = QMenu(self)

        if data.get('is_library'):
            # Library row
            lib_name = data['library']
            if self.settings.get_library_color(lib_name):
                clear_action = menu.addAction("Clear Library Color")
                clear_action.triggered.connect(lambda: self._clear_library_color(item, lib_name))
        else:
            # Pattern row
            name = data['name']
            if self.settings.get_pattern_color(name):
                clear_action = menu.addAction("Clear Pattern Color")
                clear_action.triggered.connect(lambda: self._clear_pattern_color(item, name))

        if menu.actions():
            menu.exec(self.pattern_tree.viewport().mapToGlobal(pos))

    def _clear_library_color(self, item, lib_name):
        """Clear the color for a library."""
        self.settings.set_library_color(lib_name, '')
        item.setText(3, "○")
        item.setForeground(3, QColor("#888888"))

    def _clear_pattern_color(self, item, name):
        """Clear the color for a pattern."""
        self.settings.set_pattern_color(name, '')
        item.setText(3, "○")
        item.setForeground(3, QColor("#888888"))

    def _on_selection_changed(self):
        """Handle selection change to show details."""
        items = self.pattern_tree.selectedItems()
        if not items:
            return

        data = items[0].data(0, Qt.UserRole)
        if not data or data.get('is_library'):
            return

        name = data['name']
        defn = data['defn']
        has_lua = data.get('has_lua', False)

        # Check if there's a Lua implementation with more details
        lua_info = None
        if has_lua and HAS_V3_ENGINE and hasattr(self.engine, 'lua_patterns'):
            lua_info = self.engine.lua_patterns.get(name)

        # Use display name if available, otherwise auto-generate friendly name
        if lua_info and lua_info.display_name:
            display_name = lua_info.display_name
        else:
            display_name = self._make_friendly_name(name)
        self.pattern_name_label.setText(display_name)

        # Use Lua description if available and better
        if lua_info and lua_info.description:
            self.pattern_desc_label.setText(lua_info.description)
        else:
            self.pattern_desc_label.setText(defn.get('description', 'No description'))

        self.pattern_tier_label.setText(f"Tier: {defn.get('tier', '?')}")

        # Use Lua examples if available
        examples = defn.get('examples', [])
        if lua_info and lua_info.examples:
            examples = lua_info.examples
        if examples:
            self.pattern_examples_label.setText(f"Examples: {', '.join(examples)}")
        else:
            self.pattern_examples_label.setText("Examples: (none)")

        # Use Lua odds if available
        odds = defn.get('odds', '')
        if lua_info and lua_info.odds:
            odds = lua_info.odds
        if odds:
            self.pattern_odds_label.setText(f"Odds: {odds}")
        else:
            self.pattern_odds_label.setText("Odds: (not calculated)")

        # Use Lua price if available
        price = defn.get('price_range', '')
        if lua_info and lua_info.price:
            price = lua_info.price
        if price:
            self.pattern_price_label.setText(f"Price: {price}")
        else:
            self.pattern_price_label.setText("Price: -")

        # Show threshold editor for height_ratio patterns
        rules = defn.get('rules', {})
        height_ratio_rule = None
        for rule_type in ['baseline_variance_min', 'baseline_variance_max']:
            if rule_type in rules:
                height_ratio_rule = (rule_type, rules[rule_type])
                break

        if height_ratio_rule:
            self._current_threshold_pattern = name
            rule_type, value = height_ratio_rule
            # Check for user override in SettingsManager first
            override = self.settings.get_pattern_override(name, rule_type)
            if override is not None:
                value = override
            self.threshold_label.setText(f"Threshold ({rule_type}):")
            self.threshold_edit.setText(str(value))
            self.threshold_label.show()
            self.threshold_edit.show()
            self.threshold_save_btn.show()
        else:
            self._current_threshold_pattern = None
            self.threshold_label.hide()
            self.threshold_edit.hide()
            self.threshold_save_btn.hide()

        # Show/hide Lua script viewer/editor
        if has_lua and lua_info:
            self._current_lua_pattern = name
            # Check if editable (not from core library)
            library = data.get('library', 'core')
            self._current_lua_editable = library != 'core'

            self.lua_script_label.show()
            self.view_script_btn.show()
            self.generate_serial_btn.show()
            self.generate_serial_btn.setEnabled(True)
            self.test_serial_edit.show()
            self.test_serial_edit.clear()

            # Update button text based on editability
            if self._current_lua_editable:
                self.view_script_btn.setText("Edit Script")
                self.view_script_btn.setToolTip("Edit this Lua pattern script")
                self.delete_pattern_btn.show()
            else:
                self.view_script_btn.setText("View Script")
                self.view_script_btn.setToolTip("View the Lua source code (core patterns are read-only)")
                self.delete_pattern_btn.hide()
        else:
            self._current_lua_pattern = None
            self._current_lua_editable = False
            self.lua_script_label.hide()
            self.view_script_btn.hide()
            self.delete_pattern_btn.hide()
            self.generate_serial_btn.hide()
            self.test_serial_edit.hide()

        # Update pattern preview
        self._update_pattern_preview(name, lua_info)

    def _update_pattern_preview(self, name: str, lua_info):
        """Update the pattern preview widget for the selected pattern."""
        print(f"[DEBUG] _update_pattern_preview called for: {name}")

        # Reset preview
        self.pattern_preview.set_serial("--------")
        self.pattern_preview.set_highlights([], [])
        self.pattern_preview.set_group_boxes([])
        self.match_message_label.setText("")

        # Patterns that can't generate test serials
        skip_patterns = {'GAS_PUMP', 'STAR', 'LOW_RUN_6M', 'LOW_RUN_12M', 'LOW_RUNS', 'KNOWN_SERIALS'}

        # Enable/disable button based on whether pattern can generate serials
        can_generate = name not in skip_patterns
        self.generate_serial_btn.setEnabled(can_generate)
        if not can_generate:
            self.generate_serial_btn.setToolTip("This pattern requires special conditions")
        else:
            self.generate_serial_btn.setToolTip("Generate a random matching serial")

        # Show first example if available
        examples = lua_info.examples if lua_info else []
        if examples and examples[0]:
            ex = examples[0]
            serial = f"A{ex}B" if len(ex) == 8 and ex.isdigit() else ex
            self.pattern_preview.set_serial(serial)

            # Temporarily enable pattern if disabled (for preview purposes)
            was_enabled = True
            if lua_info and name in self.engine.lua_patterns:
                was_enabled = self.engine.lua_patterns[name].enabled
                if not was_enabled:
                    self.engine.lua_patterns[name].enabled = True

            try:
                viz = self.engine.get_digit_highlights(serial, [name])
                highlights = self._flatten_highlights(viz.get('highlights', []))
                self.pattern_preview.set_highlights(highlights, viz.get('connectors', []))
                self.pattern_preview.set_group_boxes(viz.get('group_boxes', []))
            finally:
                # Restore original enabled state
                if not was_enabled and name in self.engine.lua_patterns:
                    self.engine.lua_patterns[name].enabled = False

            self.match_message_label.setText(f"Example: {serial}")

    def _flatten_highlights(self, position_highlights: list) -> list:
        """Convert per-position highlight format to simple format.

        The engine returns: [{'position': 0, 'digit': '1', 'highlights': [{'color': 'orange'}]}, ...]
        The widget expects: [{'positions': [0], 'color': 'orange'}, ...]
        """
        result = []
        for ph in position_highlights:
            pos = ph.get('position')
            for h in ph.get('highlights', []):
                result.append({
                    'positions': [pos],
                    'color': h.get('color', 'gray'),
                    'label': h.get('label', '')
                })
        return result

    def _on_test_serial_changed(self, text: str):
        """Handle manual serial input for testing against the selected pattern."""
        # Get selected pattern
        items = self.pattern_tree.selectedItems()
        if not items:
            return
        data = items[0].data(0, Qt.UserRole)
        if not data or data.get('is_library'):
            return
        name = data['name']

        # Clean up input - extract just digits
        digits = ''.join(c for c in text if c.isdigit())

        if not digits:
            # Empty input - show default example
            lua_info = self.engine.lua_patterns.get(name)
            self._update_pattern_preview(name, lua_info)
            return

        # Pad or truncate to 8 digits
        if len(digits) < 8:
            digits = digits.ljust(8, '-')
        else:
            digits = digits[:8]

        # Build full serial
        serial = f"A{digits}B"
        self.pattern_preview.set_serial(serial)

        # Check if it matches the pattern
        if '-' not in digits:
            matches = self.engine.classify_simple(serial)
            if name in matches:
                # It matches! Show highlights
                viz = self.engine.get_digit_highlights(serial, [name])
                highlights = self._flatten_highlights(viz.get('highlights', []))
                self.pattern_preview.set_highlights(highlights, viz.get('connectors', []))
                self.pattern_preview.set_group_boxes(viz.get('group_boxes', []))
                self.match_message_label.setText(f"Matches {name}")
                self.match_message_label.setStyleSheet("color: #2E7D32; font-style: italic; font-weight: bold;")
            else:
                # No match
                self.pattern_preview.set_highlights([], [])
                self.pattern_preview.set_group_boxes([])
                self.match_message_label.setText("No match")
                self.match_message_label.setStyleSheet("color: #666; font-style: italic;")
        else:
            # Incomplete serial
            self.pattern_preview.set_highlights([], [])
            self.pattern_preview.set_group_boxes([])
            self.match_message_label.setText("")
            self.match_message_label.setStyleSheet("color: #666; font-style: italic;")

    def _generate_test_serial(self):
        """Generate and display a random serial matching the selected pattern."""
        import random

        print("[DEBUG] _generate_test_serial called")

        # Clear manual test input
        self.test_serial_edit.blockSignals(True)
        self.test_serial_edit.clear()
        self.test_serial_edit.blockSignals(False)

        # Get selected pattern
        items = self.pattern_tree.selectedItems()
        if not items:
            print("[DEBUG] No items selected")
            return
        data = items[0].data(0, Qt.UserRole)
        if not data or data.get('is_library'):
            print("[DEBUG] Selected item is library or no data")
            return
        name = data['name']
        print(f"[DEBUG] Selected pattern name: {name}")

        # Temporarily enable pattern if disabled (for preview purposes)
        was_enabled = True
        if name in self.engine.lua_patterns:
            was_enabled = self.engine.lua_patterns[name].enabled
            if not was_enabled:
                self.engine.lua_patterns[name].enabled = True
                print(f"[DEBUG] Temporarily enabled pattern for preview")

        try:
            # Generate a truly random matching serial
            serial, error = self._generate_random_matching_serial(name)
            if not serial:
                self.match_message_label.setText(error or "Could not generate matching serial")
                return

            print(f"[DEBUG] Setting preview serial to: {serial}")
            self.pattern_preview.set_serial(serial)

            # Get visualization
            viz = self.engine.get_digit_highlights(serial, [name])
            raw_highlights = viz.get('highlights', [])
            connectors = viz.get('connectors', [])
            group_boxes = viz.get('group_boxes', [])

            # Count how many positions actually have highlight data
            positions_with_highlights = sum(1 for h in raw_highlights if h.get('highlights'))
            print(f"[DEBUG] Visualization: {positions_with_highlights} positions with highlights, {len(connectors)} connectors, {len(group_boxes)} group_boxes")

            highlights = self._flatten_highlights(raw_highlights)
            if highlights:
                print(f"[DEBUG]   Flattened highlights: {highlights}")
            else:
                print(f"[DEBUG]   No highlights returned by pattern!")

            self.pattern_preview.set_highlights(highlights, connectors)
            self.pattern_preview.set_group_boxes(group_boxes)
            self.match_message_label.setText(f"Generated: {serial}")
            print(f"[DEBUG] Preview update complete")
        finally:
            # Restore original enabled state
            if not was_enabled and name in self.engine.lua_patterns:
                self.engine.lua_patterns[name].enabled = False
                print(f"[DEBUG] Restored pattern to disabled state")

    def _generate_random_matching_serial(self, pattern_name: str) -> tuple:
        """Generate a random serial that matches the given pattern.

        Strategy:
        1. Get examples from Lua pattern header
        2. Analyze example structure (which positions must match)
        3. Generate new digits following that structure with random values
        4. Verify it matches, fall back to using example directly if needed

        Returns:
            tuple: (serial, None) on success, (None, error_message) on failure
        """
        import random

        prefixes = 'ABCDEFGHIJKL'
        suffixes = 'ABCDEFGHIJKLMNOPQRSTUVWXY'

        print(f"[DEBUG] Generating random serial for pattern: {pattern_name}")

        # Get pattern info and examples
        info = self.engine.lua_patterns.get(pattern_name)
        if not info or not info.examples:
            print(f"[DEBUG] No examples found for {pattern_name}")
            return None, "Pattern has no Examples defined in header"

        examples = [ex for ex in info.examples if len(ex) == 8 and ex.isdigit()]
        if not examples:
            print(f"[DEBUG] No valid 8-digit examples for {pattern_name}")
            return None, "Examples must be 8-digit numbers (e.g. \"12345678\")"

        # Special handling for patterns with mathematical constraints
        if 'SUM' in pattern_name.upper():
            digits = self._generate_sum_pattern(pattern_name, examples)
            if digits:
                prefix = random.choice(prefixes)
                suffix = random.choice(suffixes)
                serial = f"{prefix}{digits}{suffix}"
                matches = self.engine.classify_simple(serial)
                if pattern_name in matches:
                    print(f"[DEBUG] Generated sum pattern: {serial}")
                    return serial, None

        # Shuffle examples and try each one's structure
        random.shuffle(examples)

        for example in examples:
            print(f"[DEBUG] Trying example: {example}")

            # Find structure: which positions share the same digit?
            structure = self._analyze_serial_structure(example)
            print(f"[DEBUG] Structure groups: {structure}")

            # Generate new digits following the structure
            # Try up to 5 times per example to get a valid match
            for attempt in range(5):
                new_digits = self._generate_from_structure(structure, example)
                prefix = random.choice(prefixes)
                suffix = random.choice(suffixes)
                serial = f"{prefix}{new_digits}{suffix}"

                matches = self.engine.classify_simple(serial)
                if pattern_name in matches:
                    print(f"[DEBUG] Generated matching serial: {serial} (example {example}, attempt {attempt + 1})")
                    return serial, None

            # This example's structure didn't work, try next example
            print(f"[DEBUG] Structure from {example} didn't produce matches, trying next...")

        # Fallback: use a random valid example directly (with random prefix/suffix)
        print(f"[DEBUG] All structure attempts failed, finding valid example...")
        random.shuffle(examples)
        for ex in examples:
            prefix = random.choice(prefixes)
            suffix = random.choice(suffixes)
            serial = f"{prefix}{ex}{suffix}"
            matches = self.engine.classify_simple(serial)
            if pattern_name in matches:
                print(f"[DEBUG] Using example directly: {serial}")
                return serial, None

        # Last resort: return first example
        print(f"[DEBUG] No valid examples, using first anyway")
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        return f"{prefix}{examples[0]}{suffix}", None

    def _analyze_serial_structure(self, example: str) -> list:
        """Analyze which positions in the serial share the same digit.

        Returns list of position groups, where each group shares the same digit.
        Example: "12344321" -> [[0,7], [1,6], [2,5], [3,4]] (radar structure)
        Example: "11122233" -> [[0,1,2], [3,4,5], [6,7]] (triple-triple-double)
        """
        # Group positions by their digit value
        digit_positions = {}
        for i, d in enumerate(example):
            if d not in digit_positions:
                digit_positions[d] = []
            digit_positions[d].append(i)

        # Return list of position groups
        return list(digit_positions.values())

    def _generate_from_structure(self, structure: list, example: str) -> str:
        """Generate new digits following the analyzed structure.

        Each group of positions gets a random digit, but all positions
        in the same group get the same digit.
        """
        import random

        new_digits = list(example)  # Start with example as template
        used_digits = set()

        for group in structure:
            # Pick a random digit for this group
            # Try to pick a digit not used by other groups (for variety)
            available = [d for d in range(10) if d not in used_digits]
            if available:
                new_digit = random.choice(available)
            else:
                new_digit = random.randint(0, 9)

            used_digits.add(new_digit)

            # Apply to all positions in this group
            for pos in group:
                new_digits[pos] = str(new_digit)

        return ''.join(new_digits)

    def _generate_sum_pattern(self, pattern_name: str, examples: list) -> str:
        """Generate digits for sum-based patterns.

        Analyzes examples to find target sums, then generates random digits
        that add up to those targets.
        """
        import random

        # Find target sums from examples
        target_sums = set()
        for ex in examples:
            if len(ex) == 8 and ex.isdigit():
                s = sum(int(d) for d in ex)
                target_sums.add(s)

        if not target_sums:
            return None

        print(f"[DEBUG] Sum pattern targets: {target_sums}")

        # Pick a random target sum
        target = random.choice(list(target_sums))

        # Generate random digits that sum to target
        if target <= 36:  # Low sum (like 7) - start with zeros, add up
            digits = [0] * 8
            remaining = target
            while remaining > 0:
                pos = random.randint(0, 7)
                add = min(remaining, 9 - digits[pos], random.randint(1, min(3, remaining)))
                digits[pos] += add
                remaining -= add
        else:  # High sum (like 65) - start with 9s, subtract down
            digits = [9] * 8
            current_sum = 72
            while current_sum > target:
                pos = random.randint(0, 7)
                sub = min(current_sum - target, digits[pos], random.randint(1, min(3, current_sum - target)))
                digits[pos] -= sub
                current_sum -= sub

        result = ''.join(str(d) for d in digits)
        print(f"[DEBUG] Generated sum={sum(digits)} digits: {result}")
        return result

    def _save_threshold(self):
        """Save threshold override for the selected pattern."""
        if not self._current_threshold_pattern:
            return

        try:
            value = float(self.threshold_edit.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Invalid Value", "Please enter a valid number (e.g., 0.085)")
            return

        name = self._current_threshold_pattern

        # GAS_PUMP uses baseline_variance_min
        if name == 'GAS_PUMP':
            self.engine.set_gas_pump_threshold(value)
        else:
            # For other patterns, store generic threshold override
            self.settings.set_pattern_override(name, 'threshold', value)
            self.settings.save()
            self.engine.reload()

        QMessageBox.information(self, "Saved", f"Threshold for {name} set to {value}")

    def _delete_user_pattern(self):
        """Delete the currently selected user pattern after confirmation."""
        if not self._current_lua_pattern or not self._current_lua_editable:
            return

        pattern_name = self._current_lua_pattern

        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Delete Pattern",
            f"Are you sure you want to delete the pattern '{pattern_name}'?\n\n"
            "This will permanently delete the Lua file and cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Delete the pattern
        if self.engine.delete_user_pattern(pattern_name):
            QMessageBox.information(self, "Deleted", f"Pattern '{pattern_name}' has been deleted.")
            # Mark that patterns were modified
            self._patterns_modified = True
            # Reload engine to get clean state
            self.engine.reload()
            # Clear selection state
            self._current_lua_pattern = None
            self._current_lua_editable = False
            # Hide pattern action buttons
            self.lua_script_label.hide()
            self.view_script_btn.hide()
            self.delete_pattern_btn.hide()
            self.generate_serial_btn.hide()
            self.test_serial_edit.hide()
            # Clear details panel
            self.pattern_name_label.setText("Select a pattern")
            self.pattern_desc_label.setText("")
            self.pattern_tier_label.setText("Tier: -")
            self.pattern_odds_label.setText("Odds: -")
            self.pattern_price_label.setText("Price: -")
            self.pattern_examples_label.setText("Examples: -")
            self.pattern_preview.set_serial("--------")
            self.pattern_preview.set_highlights([], [])
            self.pattern_preview.set_group_boxes([])
            self.match_message_label.setText("")
            # Refresh the pattern list
            self._load_patterns()
        else:
            QMessageBox.warning(
                self,
                "Delete Failed",
                f"Could not delete pattern '{pattern_name}'.\n"
                "It may be a core pattern or the file may be protected."
            )

    def _view_lua_script(self):
        """View or edit the Lua script for the selected pattern."""
        if not self._current_lua_pattern:
            return

        if not HAS_V3_ENGINE or not hasattr(self.engine, 'lua_patterns'):
            return

        lua_info = self.engine.lua_patterns.get(self._current_lua_pattern)
        if not lua_info:
            return

        is_editable = self._current_lua_editable

        if is_editable:
            # Use CustomPatternDialog for editing user patterns
            self._edit_pattern_with_dialog(lua_info)
        else:
            # Use read-only view for core patterns
            self._view_core_pattern(lua_info)

    def _edit_pattern_with_dialog(self, lua_info):
        """Edit a user pattern using CustomPatternDialog."""
        dialog = CustomPatternDialog(
            self,
            name=lua_info.name,
            defn={
                'description': lua_info.description,
                'display_name': lua_info.display_name or '',
                'tier': lua_info.tier,
            },
            script=lua_info.script
        )

        # Store file path for saving
        dialog._edit_file_path = lua_info.file_path

        if dialog.exec() == QDialog.Accepted:
            name, defn = dialog.get_pattern()
            if name and defn.get('source') == 'lua':
                script = defn.get('script', '')

                # Validate before saving
                valid, error = self.engine.validate_script(script)
                if not valid:
                    QMessageBox.warning(
                        self, "Script Error",
                        f"Cannot save - syntax error:\n{error}"
                    )
                    return

                # Save to the existing file
                try:
                    with open(lua_info.file_path, 'w') as f:
                        f.write(script)

                    # Mark that patterns were modified
                    self._patterns_modified = True

                    # Reload the engine to pick up changes
                    self.engine.reload()
                    self._load_patterns()

                    # Find the pattern by file path (name might have changed in script header)
                    new_pattern_name = None
                    for pname, pinfo in self.engine.lua_patterns.items():
                        if pinfo.file_path == lua_info.file_path:
                            new_pattern_name = pname
                            break

                    # Re-select the pattern in the tree
                    if new_pattern_name:
                        self._select_pattern_by_name(new_pattern_name)

                    QMessageBox.information(
                        self, "Saved",
                        f"Script saved to:\n{lua_info.file_path}\n\nPatterns have been reloaded."
                    )

                except Exception as e:
                    QMessageBox.critical(self, "Save Error", f"Failed to save script:\n{e}")

    def _view_core_pattern(self, lua_info):
        """View a core (read-only) pattern."""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"View Script: {lua_info.name}")
        dialog.setMinimumSize(700, 500)

        layout = QVBoxLayout(dialog)

        # Info label
        info_text = (
            f"<b>{lua_info.name}</b> - {lua_info.description}<br>"
            f"<i>File: {lua_info.file_path}</i><br><br>"
            "Core patterns are read-only. Use 'Create Copy' to make an editable version."
        )
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Script viewer (read-only)
        script_edit = QPlainTextEdit()
        script_edit.setReadOnly(True)
        script_edit.setPlainText(lua_info.script)
        font = QFont("Consolas, Monaco, monospace")
        font.setPointSize(11)
        script_edit.setFont(font)
        script_edit.setLineWrapMode(QPlainTextEdit.NoWrap)

        # Add syntax highlighting
        highlighter = LuaSyntaxHighlighter(script_edit.document())

        layout.addWidget(script_edit)

        # Buttons
        btn_layout = QHBoxLayout()

        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(lambda: self._copy_script_to_clipboard(script_edit.toPlainText()))
        btn_layout.addWidget(copy_btn)

        create_copy_btn = QPushButton("Create Copy...")
        create_copy_btn.setToolTip("Create a new pattern based on this script")
        create_copy_btn.clicked.connect(lambda: self._create_pattern_copy(lua_info, dialog))
        btn_layout.addWidget(create_copy_btn)

        btn_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

        dialog.exec()

    def _copy_script_to_clipboard(self, script: str):
        """Copy script to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(script)
        QMessageBox.information(self, "Copied", "Script copied to clipboard!")

    def _create_pattern_copy(self, lua_info, parent_dialog):
        """Create a new pattern based on an existing Lua script."""
        # Close the view dialog
        parent_dialog.accept()

        # Generate display name from original or create new one
        original_display = lua_info.display_name if lua_info.display_name else lua_info.name.replace('_', ' ').title()

        # Open CustomPatternDialog pre-filled with the script
        dialog = CustomPatternDialog(
            self,
            name=f"{lua_info.name}_CUSTOM",
            defn={
                'description': f"Modified version of {lua_info.name}",
                'display_name': f"{original_display} (Custom)",
                'tier': lua_info.tier,
            },
            script=lua_info.script
        )

        if dialog.exec() == QDialog.Accepted:
            name, defn = dialog.get_pattern()
            if name:
                if defn.get('source') == 'lua' and HAS_V3_ENGINE:
                    self.engine.save_user_pattern(
                        name,
                        defn.get('script', ''),
                        defn.get('description', ''),
                        defn.get('tier', 5),
                        examples=defn.get('examples'),
                        display_name=defn.get('display_name', '')
                    )
                else:
                    self.engine.add_custom_pattern(name, defn)

                # Mark that patterns were modified
                self._patterns_modified = True

                # Reload patterns
                self.engine.reload()
                self._load_patterns()

                QMessageBox.information(
                    self, "Pattern Created",
                    f"Created new pattern '{name}'.\n\n"
                    "You may want to disable the original pattern if you want "
                    "only your modified version to be used."
                )
        self.engine.save_config()

    def _enable_all(self):
        """Enable all patterns and libraries."""
        for i in range(self.pattern_tree.topLevelItemCount()):
            lib_item = self.pattern_tree.topLevelItem(i)
            lib_item.setCheckState(0, Qt.Checked)  # Library checkbox
            for j in range(lib_item.childCount()):
                pattern_item = lib_item.child(j)
                pattern_item.setCheckState(2, Qt.Checked)  # Column 2 = Enabled

    def _disable_all(self):
        """Disable all patterns and libraries."""
        for i in range(self.pattern_tree.topLevelItemCount()):
            lib_item = self.pattern_tree.topLevelItem(i)
            lib_item.setCheckState(0, Qt.Unchecked)  # Library checkbox
            for j in range(lib_item.childCount()):
                pattern_item = lib_item.child(j)
                pattern_item.setCheckState(2, Qt.Unchecked)  # Column 2 = Enabled

    def _save_and_close(self):
        """Save pattern states and close."""
        # Pattern states are persisted via SettingsManager when changed
        self._save_window_geometry()
        self.settings.save()
        self.accept()

    def closeEvent(self, event):
        """Save geometry when dialog is closed."""
        self._save_window_geometry()
        self.settings.save()
        super().closeEvent(event)

    def _discover_libraries(self) -> list:
        """Discover all pattern library names (directories under patterns/)."""
        libraries = []
        patterns_dir = Path(__file__).parent.parent / 'patterns'

        if not patterns_dir.exists():
            return libraries

        for subdir in patterns_dir.iterdir():
            if subdir.is_dir() and subdir.name not in ('lib', 'data', '__pycache__'):
                libraries.append(subdir.name)

        return sorted(libraries, key=lambda x: (x != 'core', x))

    def _create_new_library(self):
        """Create a new pattern library folder."""
        name, ok = QInputDialog.getText(
            self, "New Library",
            "Enter library name:",
            text=""
        )

        if not ok or not name:
            return

        # Light sanitization - just remove characters that are problematic for file systems
        name = name.strip()
        name = ''.join(c for c in name if c not in '<>:"/\\|?*')

        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a valid library name.")
            return

        # Check if already exists
        patterns_dir = Path(__file__).parent.parent / 'patterns'
        lib_dir = patterns_dir / name

        if lib_dir.exists():
            QMessageBox.warning(self, "Already Exists", f"Library '{name}' already exists.")
            return

        # Create the directory
        try:
            lib_dir.mkdir(parents=True, exist_ok=True)
            # Create a .gitkeep to ensure the folder is tracked
            (lib_dir / '.gitkeep').touch()
            QMessageBox.information(
                self, "Library Created",
                f"Library '{name}' created at:\n{lib_dir}\n\n"
                "Add .lua pattern files to this folder, then reload patterns."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create library: {e}")

    def _open_patterns_folder(self):
        """Open the patterns directory in the system file explorer."""
        import subprocess
        import sys

        patterns_dir = Path(__file__).parent.parent / 'patterns'

        if sys.platform == 'win32':
            subprocess.Popen(['explorer', str(patterns_dir)])
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', str(patterns_dir)])
        else:
            subprocess.Popen(['xdg-open', str(patterns_dir)])

    def _show_api_docs(self):
        """Show API documentation in a popup dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Lua Pattern API Documentation")
        dialog.setMinimumSize(700, 500)

        layout = QVBoxLayout(dialog)

        # Copy button
        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(lambda: self._copy_api_docs_text())
        layout.addWidget(copy_btn)

        # Documentation
        docs = QTextEdit()
        docs.setReadOnly(True)
        docs.setHtml(self._get_api_docs_html())
        layout.addWidget(docs)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec()

    def _copy_api_docs_text(self):
        """Copy API docs as plain text to clipboard."""
        # Reuse the existing method from CustomPatternDialog
        docs = self._get_api_docs_plain_text()
        clipboard = QApplication.clipboard()
        clipboard.setText(docs)
        QMessageBox.information(self, "Copied", "API documentation copied to clipboard!")

    def _get_api_docs_html(self) -> str:
        """Get API documentation as HTML (reused from CustomPatternDialog)."""
        return '''
<h2>Pattern Script API</h2>

<h3>Script Header</h3>
<pre>
--[[
Pattern: PATTERN_NAME
DisplayName: Friendly Name With Spaces
Description: What it matches
Tier: 1-10 (1=rarest, 10=common)
Examples: ["12345678", "87654321"]
DataFile: optional_data.csv
--]]
</pre>
<p><b>DisplayName</b> is optional - if provided, it's shown in the GUI instead of the pattern name.</p>
<p><b style="color: #c9622c;">Examples</b> is required for the random preview generator to work. Without it, clicking "Generate Random" won't produce matching serials.</p>

<h3>Input Context (ctx)</h3>
<pre>
ctx.digits      -- "12345678" (8 numeric characters)
ctx.full_serial -- "A12345678B" (with prefix/suffix letters)
ctx.digit_list  -- {1,2,3,4,5,6,7,8} as integer array
ctx.metadata    -- {} additional detection data
ctx.data        -- External data (if DataFile specified)
ctx.data_by_key -- Key lookup dict (CSV only, keyed by first column)
</pre>

<h3>Return Value</h3>
<pre>
return {
    matched = true,  -- or false
    highlights = {{positions = {0, 7}, color = "orange"}},
    connectors = {{from = 0, to = 7, color = "orange", style = "arc"}},
    group_boxes = {{from = 0, to = 2, color = "gold", thickness = 3}},
    message = "Optional description"
}
</pre>

<h3>Available Colors</h3>
<p>purple, blue, cyan, orange, coral, gold, salmon, magenta, yellow, lime, teal, red, gray</p>

<h3>Connector Styles</h3>
<p>arc, line, dashed, bracket, arrow</p>

<h3>Helper Functions</h3>
<pre>
-- Analysis
count_digits(s), find_runs(s), unique_count(s), digit_sum(s), most_common(s)

-- Pattern checks
is_ladder(s), is_palindrome(s), is_repeater(s), is_alternating(s)
has_n_consecutive(s, n), all_flip_valid(s), flip_string(s)

-- String utilities
only_digits(s, allowed), starts_with(s, prefix), ends_with(s, suffix)
contains(s, substr), is_bookended(s, n)

-- Visualization helpers
highlight(positions, color, label), highlight_range(start, stop, color, label)
connector(from, to, color, style), find_digit_positions(s, digit)
</pre>

<h3>Debug Logging</h3>
<pre>
log(value1, value2, ...)  -- Log values for debugging
</pre>
<p>Use <code>log()</code> to trace script execution during batch testing. Logs appear in test results and "Copy for AI Debug" output.</p>
'''

    def _get_api_docs_plain_text(self) -> str:
        """Get API docs as plain text for clipboard."""
        return '''# Lua Pattern Script API

## Script Header
--[[
Pattern: PATTERN_NAME
DisplayName: Friendly Name With Spaces (optional)
Description: What it matches
Tier: 1-10 (1=rarest, 10=common)
Examples: ["12345678"]
DataFile: optional_data.csv
--]]

**IMPORTANT:** Examples is required for the random preview generator to work!

## Input Context (ctx)
- ctx.digits: "12345678" (8 numeric characters)
- ctx.full_serial: "A12345678B" (with prefix/suffix)
- ctx.digit_list: {1,2,3,4,5,6,7,8} as integers
- ctx.metadata: {} additional detection data
- ctx.data: External data (if DataFile specified)
- ctx.data_by_key: Key lookup dict (CSV only)

## Return Value
return {
    matched = true,
    highlights = {{positions = {0, 7}, color = "orange"}},
    connectors = {{from = 0, to = 7, color = "orange", style = "arc"}},
    group_boxes = {{from = 0, to = 2, color = "gold"}},
    message = "Description"
}

## Colors
purple, blue, cyan, orange, coral, gold, salmon, magenta, yellow, lime, teal, red, gray

## Connector Styles
arc, line, dashed, bracket, arrow

## Helper Functions
count_digits(s), find_runs(s), unique_count(s), digit_sum(s)
is_ladder(s), is_palindrome(s), is_repeater(s), is_alternating(s)
has_n_consecutive(s, n), all_flip_valid(s), flip_string(s)
highlight(positions, color), connector(from, to, color, style)

## Debug Logging
log(value1, value2, ...)  -- trace execution during testing
Logs appear in batch test results and "Copy for AI Debug" output.
'''

    def _add_custom_pattern(self):
        """Add a new custom pattern to a library."""
        # Get available libraries for the dropdown
        libraries = self._discover_libraries()
        if 'user' not in libraries:
            libraries.append('user')

        # Ask which library to add to
        default_idx = libraries.index('user') if 'user' in libraries else 0
        library, ok = QInputDialog.getItem(
            self, "Select Library",
            "Add pattern to library:",
            libraries,
            default_idx,
            False  # Not editable
        )

        if not ok:
            return

        dialog = CustomPatternDialog(self)
        if dialog.exec() == QDialog.Accepted:
            name, defn = dialog.get_pattern()
            if name:
                if defn.get('source') == 'lua' and HAS_V3_ENGINE:
                    # Save as Lua script pattern to specified library
                    self.engine.save_user_pattern(
                        name,
                        defn.get('script', ''),
                        defn.get('description', ''),
                        defn.get('tier', 5),
                        examples=defn.get('examples'),
                        library=library,
                        display_name=defn.get('display_name', '')
                    )
                else:
                    # Save as YAML rule pattern (legacy)
                    self.engine.add_custom_pattern(name, defn)

                # Mark that patterns were modified
                self._patterns_modified = True

                # Reload patterns to show the new one
                self.engine.reload()
                self._load_patterns()


class LuaSyntaxHighlighter(QSyntaxHighlighter):
    """Simple Lua syntax highlighter for QPlainTextEdit."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Keyword format
        self.keyword_format = QTextCharFormat()
        self.keyword_format.setForeground(QColor("#569CD6"))
        self.keyword_format.setFontWeight(QFont.Bold)

        # String format
        self.string_format = QTextCharFormat()
        self.string_format.setForeground(QColor("#CE9178"))

        # Comment format
        self.comment_format = QTextCharFormat()
        self.comment_format.setForeground(QColor("#6A9955"))
        self.comment_format.setFontItalic(True)

        # Number format
        self.number_format = QTextCharFormat()
        self.number_format.setForeground(QColor("#B5CEA8"))

        # Function format
        self.function_format = QTextCharFormat()
        self.function_format.setForeground(QColor("#DCDCAA"))

        # Keywords
        self.keywords = [
            'and', 'break', 'do', 'else', 'elseif', 'end', 'false',
            'for', 'function', 'if', 'in', 'local', 'nil', 'not',
            'or', 'repeat', 'return', 'then', 'true', 'until', 'while'
        ]

    def highlightBlock(self, text):
        import re

        # Keywords
        for keyword in self.keywords:
            pattern = rf'\b{keyword}\b'
            for match in re.finditer(pattern, text):
                self.setFormat(match.start(), match.end() - match.start(), self.keyword_format)

        # Strings (single and double quotes)
        for match in re.finditer(r'"[^"]*"', text):
            self.setFormat(match.start(), match.end() - match.start(), self.string_format)
        for match in re.finditer(r"'[^']*'", text):
            self.setFormat(match.start(), match.end() - match.start(), self.string_format)

        # Numbers
        for match in re.finditer(r'\b\d+\.?\d*\b', text):
            self.setFormat(match.start(), match.end() - match.start(), self.number_format)

        # Single-line comments
        for match in re.finditer(r'--.*$', text):
            self.setFormat(match.start(), match.end() - match.start(), self.comment_format)

        # Function calls
        for match in re.finditer(r'\b(\w+)\s*\(', text):
            self.setFormat(match.start(1), match.end(1) - match.start(1), self.function_format)


class DigitPreviewWidget(QWidget):
    """Widget showing digit boxes with highlights and connectors."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.serial = "A--------B"  # Full serial with prefix/suffix
        self.highlights = []
        self.connectors = []
        self.group_boxes = []
        self.setMinimumHeight(100)
        self.setMinimumWidth(550)

    def set_serial(self, serial: str):
        """Set the serial number to display."""
        # Keep dashes as placeholder
        if serial == "--------":
            self.serial = "A--------B"
        elif len(serial) == 8:
            # Just digits, add default prefix/suffix
            self.serial = f"A{serial}B"
        elif len(serial) >= 10:
            # Full serial with prefix/suffix
            self.serial = serial[:10]
        else:
            # Pad to 10 characters
            self.serial = serial.ljust(10, '-')[:10]
        self.update()

    def set_highlights(self, highlights: list, connectors: list):
        """Set highlights and connectors."""
        self.highlights = highlights or []
        self.connectors = connectors or []
        self.update()

    def set_group_boxes(self, group_boxes: list):
        """Set group boxes for multi-digit highlighting."""
        self.group_boxes = group_boxes or []
        self.update()

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QPen, QBrush, QPainterPath
        from PySide6.QtCore import QRect, QPoint

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Bill-like colors
        bg_color = QColor("#D4C4A8")  # Tan/cream - bill paper
        digit_color = QColor("#1B5E20")  # Dark green - bill ink
        border_color = QColor("#5D4037")  # Brown border

        # Fill background with tan color
        painter.fillRect(self.rect(), bg_color)

        # Calculate box dimensions - 1.5x size with tighter boxes
        box_width = 48
        box_height = 63
        spacing = 9
        num_chars = 10  # prefix + 8 digits + suffix
        total_width = num_chars * box_width + (num_chars - 1) * spacing
        start_x = (self.width() - total_width) // 2
        start_y = 18

        # Build color map for each digit position (0-7 maps to character position 1-8)
        position_colors = {}
        for h in self.highlights:
            positions = h.get('positions', [])
            color = h.get('color', 'gray')
            for pos in positions:
                if 0 <= pos < 8:
                    # Offset by 1 for prefix letter
                    position_colors[pos + 1] = color

        # Color mapping
        color_map = {
            'purple': QColor("#9C27B0"),
            'blue': QColor("#2196F3"),
            'cyan': QColor("#00BCD4"),
            'orange': QColor("#FF9800"),
            'coral': QColor("#FF7043"),
            'gold': QColor("#FFD700"),
            'salmon': QColor("#FA8072"),
            'magenta': QColor("#E91E63"),
            'yellow': QColor("#FFEB3B"),
            'lime': QColor("#8BC34A"),
            'teal': QColor("#009688"),
            'red': QColor("#F44336"),
            'gray': QColor("#9E9E9E"),
        }

        # Draw characters (prefix letter + 8 digits + suffix letter)
        font = painter.font()
        font.setPointSize(33)
        font.setBold(True)
        painter.setFont(font)

        box_rects = []  # Only store digit box rects (indices 1-8)
        for i, char in enumerate(self.serial):
            x = start_x + i * (box_width + spacing)
            rect = QRect(x, start_y, box_width, box_height)

            is_letter = (i == 0 or i == 9)  # First and last are letters

            if is_letter:
                # Letters: no box, just draw the character
                painter.setPen(QPen(digit_color))
                painter.drawText(rect, Qt.AlignCenter, char)
                box_rects.append(None)  # Placeholder for indexing
            else:
                box_rects.append(rect)

                # Background and border for each digit box
                if i in position_colors:
                    # Highlighted digit - use highlight color for border
                    hl_color = color_map.get(position_colors[i], QColor("#9E9E9E"))
                    painter.setBrush(QBrush(bg_color.lighter(105)))
                    painter.setPen(QPen(hl_color, 3))
                    painter.drawRoundedRect(rect, 5, 5)

                # Draw digit in green
                painter.setPen(QPen(digit_color))
                painter.drawText(rect, Qt.AlignCenter, char)

        # Draw group boxes (spans multiple digits with a single box)
        # Note: digit positions 0-7 map to box_rects indices 1-8 (offset for prefix letter)
        for gb in self.group_boxes:
            from_pos = gb.get('from', 0)
            to_pos = gb.get('to', 0)
            color = gb.get('color', 'gold')
            thickness = gb.get('thickness', 3)

            if 0 <= from_pos < 8 and 0 <= to_pos < 8 and from_pos <= to_pos:
                from_rect = box_rects[from_pos + 1]  # +1 for prefix letter offset
                to_rect = box_rects[to_pos + 1]

                if from_rect and to_rect:
                    # Create spanning rectangle with some padding
                    padding = 4
                    span_rect = QRect(
                        from_rect.left() - padding,
                        from_rect.top() - padding,
                        to_rect.right() - from_rect.left() + 2 * padding,
                        from_rect.height() + 2 * padding
                    )

                    pen = QPen(color_map.get(color, QColor("#FFD700")), thickness)
                    painter.setPen(pen)
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRoundedRect(span_rect, 8, 8)

        # Draw connectors (arcs above the boxes)
        # Note: digit positions 0-7 map to box_rects indices 1-8 (offset for prefix letter)
        for conn in self.connectors:
            from_pos = conn.get('from', 0)
            to_pos = conn.get('to', 0)
            color = conn.get('color', 'gray')
            style = conn.get('style', 'arc')

            if 0 <= from_pos < 8 and 0 <= to_pos < 8:
                from_rect = box_rects[from_pos + 1]  # +1 for prefix letter offset
                to_rect = box_rects[to_pos + 1]

                if from_rect and to_rect:
                    from_x = from_rect.center().x()
                    to_x = to_rect.center().x()
                    y = start_y - 5

                    pen = QPen(color_map.get(color, QColor("#9E9E9E")), 2)
                    if style == 'dashed':
                        pen.setStyle(Qt.DashLine)
                    painter.setPen(pen)

                    # Draw arc
                    mid_x = (from_x + to_x) // 2
                    arc_height = min(20, abs(to_pos - from_pos) * 5)

                    path = QPainterPath()
                    path.moveTo(from_x, y)
                    path.quadTo(mid_x, y - arc_height, to_x, y)
                    painter.drawPath(path)

        painter.end()


class CustomPatternDialog(QDialog):
    """Dialog for adding/editing a custom pattern (simple rule or Lua script)."""

    def __init__(self, parent=None, name: str = "", defn: dict = None, script: str = None):
        super().__init__(parent)
        self.setWindowTitle("Add Custom Pattern" if not name else "Edit Custom Pattern")
        self.setMinimumSize(700, 800)

        self.original_name = name
        self.defn = defn or {}
        self.original_script = script
        self.is_lua_pattern = script is not None

        # Try to get the v3 engine for Lua support
        self.engine = None
        if HAS_V3_ENGINE:
            try:
                self.engine = PatternEngine()
            except Exception:
                pass

        # Batch testing instance variables
        self.should_match_edit: QLineEdit = None
        self.should_not_match_edit: QLineEdit = None
        self.copy_debug_btn: QPushButton = None
        self._last_batch_results: list = []
        self._last_script: str = ""

        # Wizard mode instance variables
        self._is_wizard_pattern: bool = False
        self._wizard_generated_script: str = ""
        self._wizard_examples: list = []
        self.param_widgets: dict = {}  # Will be populated by _create_wizard_tab

        self._setup_ui()
        if name:
            self._load_existing()
        # Delay resize until after dialog is fully constructed
        QTimer.singleShot(0, lambda: self.resize(700, 800))

    def _setup_ui(self):
        """Setup the dialog UI with tabs."""
        layout = QVBoxLayout(self)

        # Header - pattern name and description (always visible)
        header_group = QGroupBox("Pattern Info")
        header_layout = QFormLayout(header_group)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., MY_BIRTHDAY, CUSTOM_RADAR")
        self.name_edit.textChanged.connect(self._auto_generate_display_name)
        header_layout.addRow("Pattern Name:", self.name_edit)

        self.display_name_edit = QLineEdit()
        self.display_name_edit.setPlaceholderText("e.g., My Birthday, Custom Radar (shown in GUI)")
        header_layout.addRow("Display Name:", self.display_name_edit)

        self.desc_edit = QLineEdit()
        self.desc_edit.setPlaceholderText("Brief description of what this pattern matches")
        header_layout.addRow("Description:", self.desc_edit)

        tier_layout = QHBoxLayout()
        self.tier_spin = QSpinBox()
        self.tier_spin.setRange(1, 10)
        self.tier_spin.setValue(5)
        tier_layout.addWidget(self.tier_spin)
        tier_layout.addWidget(QLabel("(1=rare/valuable, 10=common/novelty)"))
        tier_layout.addStretch()
        header_layout.addRow("Tier:", tier_layout)

        layout.addWidget(header_group)

        # Tab widget for Simple vs Script modes
        self.tab_widget = QTabWidget()

        # Tab 1: Simple Rules (YAML-based)
        self.simple_tab = self._create_simple_tab()
        self.tab_widget.addTab(self.simple_tab, "Simple Rule")

        # Tab 2: Pattern Wizard
        if HAS_V3_ENGINE:
            self.wizard_tab = self._create_wizard_tab()
            self.tab_widget.addTab(self.wizard_tab, "Pattern Wizard")

        # Tab 3: AI Generate
        if HAS_V3_ENGINE:
            self.ai_tab = self._create_ai_tab()
            self.tab_widget.addTab(self.ai_tab, "AI Generate")

        # Tab 4: Lua Script (if v3 engine available)
        if HAS_V3_ENGINE:
            self.script_tab = self._create_script_tab()
            self.tab_widget.addTab(self.script_tab, "Lua Script")

            # Tab 5: Test/Preview
            self.test_tab = self._create_test_tab()
            self.tab_widget.addTab(self.test_tab, "Test")

            # Tab 6: Documentation
            self.docs_tab = self._create_docs_tab()
            self.tab_widget.addTab(self.docs_tab, "API Docs")

        layout.addWidget(self.tab_widget)

        # Buttons
        button_layout = QHBoxLayout()

        if HAS_V3_ENGINE:
            validate_btn = QPushButton("Validate Script")
            validate_btn.clicked.connect(self._validate_script)
            button_layout.addWidget(validate_btn)

        button_layout.addStretch()

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._validate_and_accept)
        button_box.rejected.connect(self.reject)
        button_layout.addWidget(button_box)

        layout.addLayout(button_layout)

    def _create_simple_tab(self) -> QWidget:
        """Create the simple rules tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        form = QFormLayout()

        # Rule type
        self.rule_type = QComboBox()
        self.rule_type.addItems([
            "contains", "starts_with", "ends_with", "regex",
            "baseline_variance_min", "baseline_variance_max"
        ])
        self.rule_type.currentTextChanged.connect(self._update_hint)
        form.addRow("Rule Type:", self.rule_type)

        # Value
        self.value_edit = QLineEdit()
        self.value_edit.setPlaceholderText("e.g., 0704 for July 4th")
        form.addRow("Value:", self.value_edit)

        # Hint
        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("color: gray; font-style: italic;")
        self._update_hint()
        form.addRow("", self.hint_label)

        layout.addLayout(form)
        layout.addStretch()

        return widget

    def _create_wizard_tab(self) -> QWidget:
        """Create the Pattern Wizard tab for recipe-based pattern creation."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Recipe selector
        recipe_group = QGroupBox("Recipe Type")
        recipe_layout = QVBoxLayout(recipe_group)

        self.recipe_combo = QComboBox()
        self.recipes = get_all_recipes()
        for recipe in self.recipes:
            self.recipe_combo.addItem(recipe.display_name)
        self.recipe_combo.currentIndexChanged.connect(self._on_recipe_changed)
        recipe_layout.addWidget(self.recipe_combo)

        # Recipe description
        self.recipe_desc_label = QLabel()
        self.recipe_desc_label.setWordWrap(True)
        self.recipe_desc_label.setStyleSheet("color: gray; font-style: italic;")
        recipe_layout.addWidget(self.recipe_desc_label)

        layout.addWidget(recipe_group)

        # Parameters section (dynamic)
        self.params_group = QGroupBox("Parameters")
        self.params_layout = QVBoxLayout(self.params_group)
        self.param_widgets = {}  # name -> widget

        layout.addWidget(self.params_group)

        # Visualization options
        viz_group = QGroupBox("Visualization")
        viz_layout = QFormLayout(viz_group)

        self.wizard_color_combo = QComboBox()
        self.wizard_color_combo.addItems([
            "orange", "lime", "cyan", "blue", "purple", "coral",
            "gold", "salmon", "magenta", "yellow", "teal", "red", "gray"
        ])
        self.wizard_color_combo.currentTextChanged.connect(self._on_wizard_param_changed)
        viz_layout.addRow("Highlight color:", self.wizard_color_combo)

        layout.addWidget(viz_group)

        # Live preview
        preview_group = QGroupBox("Live Preview")
        preview_layout = QVBoxLayout(preview_group)

        # Example serials
        self.wizard_examples_label = QLabel("Matching examples will appear here")
        self.wizard_examples_label.setWordWrap(True)
        preview_layout.addWidget(self.wizard_examples_label)

        # Digit preview widget
        self.wizard_preview = DigitPreviewWidget()
        self.wizard_preview.setMinimumHeight(100)
        preview_layout.addWidget(self.wizard_preview)

        # Preview message
        self.wizard_preview_message = QLabel("")
        self.wizard_preview_message.setWordWrap(True)
        self.wizard_preview_message.setStyleSheet("color: #666; font-style: italic;")
        preview_layout.addWidget(self.wizard_preview_message)

        layout.addWidget(preview_group)

        # Generated code section (collapsible)
        self.code_group = QGroupBox("Generated Lua Code")
        self.code_group.setCheckable(True)
        self.code_group.setChecked(False)
        code_layout = QVBoxLayout(self.code_group)

        self.wizard_code_preview = QPlainTextEdit()
        self.wizard_code_preview.setReadOnly(True)
        self.wizard_code_preview.setMaximumHeight(200)
        font = QFont("Consolas, Monaco, monospace")
        font.setPointSize(10)
        self.wizard_code_preview.setFont(font)
        code_layout.addWidget(self.wizard_code_preview)

        # Connect checkbox to show/hide content
        self.code_group.toggled.connect(self._on_code_group_toggled)

        layout.addWidget(self.code_group)

        # Initialize with first recipe
        self._on_recipe_changed(0)

        return widget

    def _on_code_group_toggled(self, checked: bool):
        """Show/hide generated code content."""
        self.wizard_code_preview.setVisible(checked)
        if checked:
            self._update_wizard_preview()

    def _on_recipe_changed(self, index: int):
        """Handle recipe type change."""
        if index < 0 or index >= len(self.recipes):
            return

        recipe = self.recipes[index]
        self.recipe_desc_label.setText(recipe.description)

        # Clear existing parameter widgets
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

        self.param_widgets = {}

        # Create widgets for each parameter
        for param_def in recipe.get_parameter_definitions():
            self._create_param_widget(param_def)

        # Update preview
        self._update_wizard_preview()

    def _clear_layout(self, layout):
        """Recursively clear a layout."""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

    def _create_param_widget(self, param_def: ParameterDef):
        """Create a widget for a parameter definition."""
        row_layout = QHBoxLayout()

        label = QLabel(param_def.label + ":")
        label.setMinimumWidth(120)
        row_layout.addWidget(label)

        if param_def.widget_type == "dropdown":
            widget = QComboBox()
            widget.addItems(param_def.options)
            if param_def.default:
                idx = widget.findText(str(param_def.default))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            widget.currentTextChanged.connect(self._on_wizard_param_changed)
            self.param_widgets[param_def.name] = widget
            row_layout.addWidget(widget)

        elif param_def.widget_type == "spinbox":
            widget = QSpinBox()
            if param_def.min_value is not None:
                widget.setMinimum(param_def.min_value)
            if param_def.max_value is not None:
                widget.setMaximum(param_def.max_value)
            if param_def.default is not None:
                widget.setValue(param_def.default)
            widget.valueChanged.connect(self._on_wizard_param_changed)
            self.param_widgets[param_def.name] = widget
            row_layout.addWidget(widget)

        elif param_def.widget_type == "checkbox_group":
            # Create a horizontal layout with checkboxes
            check_layout = QVBoxLayout()

            # First row: checkboxes for each option
            checkbox_row = QHBoxLayout()
            checkboxes = {}
            for opt in param_def.options:
                cb = QCheckBox(opt)
                if param_def.default and opt in param_def.default:
                    cb.setChecked(True)
                cb.stateChanged.connect(self._on_wizard_param_changed)
                checkbox_row.addWidget(cb)
                checkboxes[opt] = cb
            check_layout.addLayout(checkbox_row)

            # Second row: preset buttons
            preset_row = QHBoxLayout()
            presets = {
                "Binary (0,1)": ["0", "1"],
                "Flipper (0,1,6,8,9)": ["0", "1", "6", "8", "9"],
                "Evens (0,2,4,6,8)": ["0", "2", "4", "6", "8"],
                "Odds (1,3,5,7,9)": ["1", "3", "5", "7", "9"],
            }
            for preset_name, preset_digits in presets.items():
                btn = QPushButton(preset_name.split(" ")[0])  # Just "Binary", "Flipper", etc.
                btn.setMaximumWidth(70)
                btn.clicked.connect(
                    lambda checked, cbs=checkboxes, digits=preset_digits:
                    self._apply_digit_preset(cbs, digits)
                )
                preset_row.addWidget(btn)
            preset_row.addStretch()
            check_layout.addLayout(preset_row)

            self.param_widgets[param_def.name] = checkboxes
            row_layout.addLayout(check_layout)

        elif param_def.widget_type == "radio":
            # Create radio buttons
            radio_layout = QHBoxLayout()
            from PySide6.QtWidgets import QButtonGroup, QRadioButton
            group = QButtonGroup(self)
            radios = {}
            for i, opt in enumerate(param_def.options):
                rb = QRadioButton(opt)
                if param_def.default and opt == param_def.default:
                    rb.setChecked(True)
                elif i == 0 and not param_def.default:
                    rb.setChecked(True)
                rb.toggled.connect(self._on_wizard_param_changed)
                group.addButton(rb)
                radio_layout.addWidget(rb)
                radios[opt] = rb
            self.param_widgets[param_def.name] = radios
            row_layout.addLayout(radio_layout)

        row_layout.addStretch()

        # Add description as tooltip on the label
        if param_def.description:
            label.setToolTip(param_def.description)

        self.params_layout.addLayout(row_layout)

    def _apply_digit_preset(self, checkboxes: dict, digits: list):
        """Apply a digit preset to checkboxes."""
        for digit, cb in checkboxes.items():
            cb.setChecked(digit in digits)

    def _on_wizard_param_changed(self, *args):
        """Handle parameter change in wizard."""
        self._update_wizard_preview()

    def _get_wizard_params(self) -> dict:
        """Get current parameter values from wizard widgets."""
        params = {}
        for name, widget in self.param_widgets.items():
            if isinstance(widget, QComboBox):
                params[name] = widget.currentText()
            elif isinstance(widget, QSpinBox):
                params[name] = widget.value()
            elif isinstance(widget, dict):
                # Checkbox group or radio group
                first_widget = next(iter(widget.values()))
                if isinstance(first_widget, QCheckBox):
                    # Checkbox group - return list of checked values
                    params[name] = [k for k, cb in widget.items() if cb.isChecked()]
                else:
                    # Radio group - return selected value
                    for k, rb in widget.items():
                        if rb.isChecked():
                            params[name] = k
                            break
        return params

    def _get_current_recipe(self):
        """Get the currently selected recipe."""
        index = self.recipe_combo.currentIndex()
        if 0 <= index < len(self.recipes):
            return self.recipes[index]
        return None

    def _update_wizard_preview(self):
        """Update the wizard preview with current settings."""
        recipe = self._get_current_recipe()
        if not recipe:
            return

        params = self._get_wizard_params()
        color = self.wizard_color_combo.currentText()

        # Generate examples
        try:
            examples = recipe.generate_examples(params)
            if examples:
                self.wizard_examples_label.setText("Examples: " + ", ".join(examples[:5]))

                # Show first example in preview
                test_serial = examples[0]

                # Generate Lua and test it
                pattern_name = self.name_edit.text().strip().upper().replace(' ', '_') or "WIZARD_PATTERN"
                description = self.desc_edit.text().strip() or recipe.description
                tier = self.tier_spin.value()

                lua_code = recipe.generate_lua(params, pattern_name, description, tier, color)

                # Update code preview
                self.wizard_code_preview.setPlainText(lua_code)

                # Test the pattern
                if self.engine:
                    try:
                        result = self.engine.test_script(lua_code, f"A{test_serial}B")
                        if result and result.matched:
                            self.wizard_preview.set_serial(
                                test_serial,
                                result.highlights or [],
                                result.connectors or [],
                                result.group_boxes or []
                            )
                            self.wizard_preview_message.setText(result.message or 'Pattern matched')
                            self.wizard_preview_message.setStyleSheet("color: #2e7d32; font-weight: bold;")
                        else:
                            self.wizard_preview.set_serial(test_serial, [], [], [])
                            self.wizard_preview_message.setText("Pattern did not match (check parameters)")
                            self.wizard_preview_message.setStyleSheet("color: #d32f2f;")
                    except Exception as e:
                        self.wizard_preview.set_serial(test_serial, [], [], [])
                        self.wizard_preview_message.setText(f"Error: {str(e)[:50]}")
                        self.wizard_preview_message.setStyleSheet("color: #d32f2f;")
            else:
                self.wizard_examples_label.setText("No examples could be generated")
                self.wizard_preview_message.setText("")
        except Exception as e:
            self.wizard_examples_label.setText(f"Error: {str(e)[:50]}")
            self.wizard_preview_message.setText("")

    def _create_ai_tab(self) -> QWidget:
        """Create the AI Generate tab for AI-assisted pattern creation."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Description input
        desc_group = QGroupBox("Describe Your Pattern")
        desc_layout = QVBoxLayout(desc_group)

        desc_hint = QLabel("Describe the pattern you want to create in plain English:")
        desc_hint.setStyleSheet("color: gray;")
        desc_layout.addWidget(desc_hint)

        self.ai_description_edit = QTextEdit()
        self.ai_description_edit.setPlaceholderText(
            "Example: Find serials where there are exactly 3 pairs of consecutive identical digits "
            "(like 11, 22, 33) anywhere in the serial, but not 4 pairs."
        )
        self.ai_description_edit.setMaximumHeight(100)
        desc_layout.addWidget(self.ai_description_edit)

        layout.addWidget(desc_group)

        # Optional examples
        examples_group = QGroupBox("Optional: Provide Examples")
        examples_layout = QFormLayout(examples_group)

        self.ai_should_match_edit = QLineEdit()
        self.ai_should_match_edit.setPlaceholderText("11223456, 00112345, 12334455")
        examples_layout.addRow("Should match:", self.ai_should_match_edit)

        self.ai_should_not_match_edit = QLineEdit()
        self.ai_should_not_match_edit.setPlaceholderText("11223344, 12345678, 00000000")
        examples_layout.addRow("Should NOT match:", self.ai_should_not_match_edit)

        layout.addWidget(examples_group)

        # Generate button and status
        generate_layout = QHBoxLayout()

        self.ai_generate_btn = QPushButton("Generate Pattern")
        self.ai_generate_btn.setMinimumHeight(35)
        self.ai_generate_btn.clicked.connect(self._on_ai_generate)
        generate_layout.addWidget(self.ai_generate_btn)

        generate_layout.addStretch()

        self.ai_status_label = QLabel("")
        self.ai_status_label.setStyleSheet("color: gray;")
        generate_layout.addWidget(self.ai_status_label)

        layout.addLayout(generate_layout)

        # Generated code preview
        code_group = QGroupBox("Generated Code")
        code_layout = QVBoxLayout(code_group)

        self.ai_code_preview = QPlainTextEdit()
        self.ai_code_preview.setReadOnly(True)
        font = QFont("Consolas, Monaco, monospace")
        font.setPointSize(10)
        self.ai_code_preview.setFont(font)
        self.ai_code_preview.setPlaceholderText("Generated Lua code will appear here...")
        code_layout.addWidget(self.ai_code_preview)

        # Action buttons
        action_layout = QHBoxLayout()

        self.ai_use_code_btn = QPushButton("Use This Code")
        self.ai_use_code_btn.setToolTip("Copy the generated code to the Lua Script tab")
        self.ai_use_code_btn.setEnabled(False)
        self.ai_use_code_btn.clicked.connect(self._on_ai_use_code)
        action_layout.addWidget(self.ai_use_code_btn)

        self.ai_test_code_btn = QPushButton("Test Code")
        self.ai_test_code_btn.setToolTip("Switch to the Test tab to test the generated code")
        self.ai_test_code_btn.setEnabled(False)
        self.ai_test_code_btn.clicked.connect(self._on_ai_test_code)
        action_layout.addWidget(self.ai_test_code_btn)

        action_layout.addStretch()

        code_layout.addLayout(action_layout)
        layout.addWidget(code_group)

        # Configuration hint
        config_hint = QLabel("Configure your AI provider in Settings → AI tab")
        config_hint.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(config_hint)

        return widget

    def _on_ai_generate(self):
        """Handle AI generate button click."""
        from settings_manager import get_settings

        settings = get_settings()

        # Check if AI is configured
        if not settings.ai.provider:
            QMessageBox.warning(
                self,
                "AI Not Configured",
                "Please configure your AI provider in Settings → AI tab first."
            )
            return

        if not settings.ai.api_key:
            QMessageBox.warning(
                self,
                "API Key Missing",
                "Please enter your API key in Settings → AI tab."
            )
            return

        description = self.ai_description_edit.toPlainText().strip()
        if not description:
            QMessageBox.warning(
                self,
                "Description Required",
                "Please enter a description of the pattern you want to create."
            )
            return

        # Parse examples
        should_match = self._parse_example_serials(self.ai_should_match_edit.text())
        should_not_match = self._parse_example_serials(self.ai_should_not_match_edit.text())

        # Get suggested pattern name from header
        pattern_name = self.name_edit.text().strip().upper().replace(' ', '_')

        # Determine model
        if settings.ai.provider == "anthropic":
            model = settings.ai.anthropic_model
        else:
            model = settings.ai.openai_model

        # Create generator
        generator = AIPatternGenerator(
            provider=settings.ai.provider,
            api_key=settings.ai.api_key,
            model=model
        )

        # Update UI
        self.ai_generate_btn.setEnabled(False)
        self.ai_status_label.setText("Generating...")
        self.ai_status_label.setStyleSheet("color: blue;")
        self.ai_code_preview.setPlainText("")
        self.ai_use_code_btn.setEnabled(False)
        self.ai_test_code_btn.setEnabled(False)

        # Force UI update
        QApplication.processEvents()

        # Generate
        def progress_callback(msg):
            self.ai_status_label.setText(msg)
            QApplication.processEvents()

        result = generator.generate(
            description=description,
            should_match=should_match,
            should_not_match=should_not_match,
            pattern_name=pattern_name,
            progress_callback=progress_callback
        )

        self.ai_generate_btn.setEnabled(True)

        if result.success:
            self.ai_code_preview.setPlainText(result.lua_code)
            self.ai_status_label.setText("✓ Generated successfully")
            self.ai_status_label.setStyleSheet("color: green;")
            self.ai_use_code_btn.setEnabled(True)
            self.ai_test_code_btn.setEnabled(True)

            # Store for later use
            self._ai_generated_code = result.lua_code
        else:
            self.ai_status_label.setText(f"✗ {result.error}")
            self.ai_status_label.setStyleSheet("color: red;")

            if result.raw_response:
                # Show raw response for debugging
                self.ai_code_preview.setPlainText(
                    f"-- Error: {result.error}\n"
                    f"-- Raw response:\n{result.raw_response}"
                )

    def _on_ai_use_code(self):
        """Copy generated code to Lua Script tab."""
        if hasattr(self, '_ai_generated_code') and self._ai_generated_code:
            self.script_edit.setPlainText(self._ai_generated_code)
            # Switch to Lua Script tab (index 3, after AI Generate)
            self.tab_widget.setCurrentIndex(3)
            self.ai_status_label.setText("Code copied to Lua Script tab")

    def _on_ai_test_code(self):
        """Copy code to Lua Script tab and switch to Test tab."""
        if hasattr(self, '_ai_generated_code') and self._ai_generated_code:
            self.script_edit.setPlainText(self._ai_generated_code)
            # Switch to Test tab (index 4)
            self.tab_widget.setCurrentIndex(4)

    def _parse_example_serials(self, text: str) -> list[str]:
        """Parse comma-separated serial numbers."""
        serials = []
        for entry in text.strip().split(','):
            entry = entry.strip()
            if not entry:
                continue
            # Extract digits only
            digits = ''.join(c for c in entry if c.isdigit())
            if len(digits) == 8:
                serials.append(digits)
            elif len(digits) == 10 and len(entry) >= 10:
                # Full serial like A12345678B - extract middle 8
                serials.append(digits[:8] if entry[0].isalpha() else digits[-8:])
        return serials

    def _create_script_tab(self) -> QWidget:
        """Create the Lua script editor tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Toolbar
        toolbar = QHBoxLayout()

        format_btn = QPushButton("Format")
        format_btn.clicked.connect(self._format_script)
        toolbar.addWidget(format_btn)

        template_combo = QComboBox()
        template_combo.addItem("-- Select Template --")
        template_combo.addItem("Basic Match")
        template_combo.addItem("Palindrome Check")
        template_combo.addItem("Run Detection")
        template_combo.addItem("Digit Count")
        template_combo.currentTextChanged.connect(self._insert_template)
        toolbar.addWidget(template_combo)

        toolbar.addStretch()

        layout.addLayout(toolbar)

        # Code editor
        self.script_edit = QPlainTextEdit()
        font = QFont("Consolas, Monaco, monospace")
        font.setPointSize(11)
        self.script_edit.setFont(font)
        self.script_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.script_edit.setTabStopDistance(QFontMetrics(font).horizontalAdvance(' ') * 4)

        # Add syntax highlighter
        self.highlighter = LuaSyntaxHighlighter(self.script_edit.document())

        # Default template
        default_script = '''function match(ctx)
    -- ctx.digits: "12345678" (8 numeric characters)
    -- ctx.full_serial: "A12345678B" (with prefix/suffix)
    -- ctx.digit_list: {1,2,3,4,5,6,7,8} as integers

    -- Your matching logic here
    if ctx.digits == "12345678" then
        return {
            matched = true,
            highlights = {
                {positions = {0, 1, 2, 3}, color = "orange"},
            },
            connectors = {},
            message = "Custom match!"
        }
    end

    return {matched = false}
end
'''
        self.script_edit.setPlainText(default_script)
        layout.addWidget(self.script_edit)

        # Status bar
        self.script_status = QLabel("Ready")
        self.script_status.setStyleSheet("color: gray;")
        layout.addWidget(self.script_status)

        return widget

    def _create_test_tab(self) -> QWidget:
        """Create the test/preview tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Quick Test group
        quick_group = QGroupBox("Quick Test")
        quick_layout = QVBoxLayout(quick_group)

        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Test Serial:"))

        self.test_serial_edit = QLineEdit()
        self.test_serial_edit.setPlaceholderText("e.g., A12344321B or just 12344321")
        self.test_serial_edit.setText("A12344321B")
        self.test_serial_edit.textChanged.connect(self._run_live_test)
        input_layout.addWidget(self.test_serial_edit)

        test_btn = QPushButton("Test")
        test_btn.clicked.connect(self._run_live_test)
        input_layout.addWidget(test_btn)

        quick_layout.addLayout(input_layout)
        layout.addWidget(quick_group)

        # Preview widget
        preview_group = QGroupBox("Visual Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.digit_preview = DigitPreviewWidget()
        preview_layout.addWidget(self.digit_preview)

        layout.addWidget(preview_group)

        # Batch Test Cases group
        batch_group = QGroupBox("Batch Test Cases")
        batch_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        batch_layout = QVBoxLayout(batch_group)

        # Stacked layout for should match / should not match (comma-delimited)
        # Labels above inputs for full width
        batch_layout.addWidget(QLabel("Should Match:"))
        self.should_match_edit = QLineEdit()
        self.should_match_edit.setPlaceholderText("12344321, 11111111, 45677654")
        self.should_match_edit.setMinimumHeight(25)
        batch_layout.addWidget(self.should_match_edit)

        batch_layout.addSpacing(8)

        batch_layout.addWidget(QLabel("Should NOT Match:"))
        self.should_not_match_edit = QLineEdit()
        self.should_not_match_edit.setPlaceholderText("12345678, 12344322, 98765432")
        self.should_not_match_edit.setMinimumHeight(25)
        batch_layout.addWidget(self.should_not_match_edit)

        batch_layout.addSpacing(8)

        # Batch buttons row
        batch_btn_layout = QHBoxLayout()
        run_batch_btn = QPushButton("Run All Tests")
        run_batch_btn.setMinimumHeight(28)
        run_batch_btn.clicked.connect(self._run_batch_tests)
        batch_btn_layout.addWidget(run_batch_btn)

        export_cases_btn = QPushButton("Export for AI")
        export_cases_btn.setMinimumHeight(28)
        export_cases_btn.setToolTip("Copy test cases formatted for AI prompts")
        export_cases_btn.clicked.connect(self._export_test_cases)
        batch_btn_layout.addWidget(export_cases_btn)

        batch_btn_layout.addStretch()
        batch_layout.addLayout(batch_btn_layout)

        layout.addWidget(batch_group)

        # Results
        results_group = QGroupBox("Test Results")
        results_layout = QVBoxLayout(results_group)

        self.test_results = QTextEdit()
        self.test_results.setReadOnly(True)
        self.test_results.setMaximumHeight(150)
        results_layout.addWidget(self.test_results)

        # Copy for AI Debug button
        debug_btn_layout = QHBoxLayout()
        self.copy_debug_btn = QPushButton("Copy for AI Debug")
        self.copy_debug_btn.setToolTip("Copy script + failing tests for AI assistance")
        self.copy_debug_btn.setEnabled(False)
        self.copy_debug_btn.clicked.connect(self._copy_debug_info)
        debug_btn_layout.addWidget(self.copy_debug_btn)
        debug_btn_layout.addStretch()
        results_layout.addLayout(debug_btn_layout)

        layout.addWidget(results_group)

        return widget

    def _create_docs_tab(self) -> QWidget:
        """Create the API documentation tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Button row with both copy buttons
        btn_layout = QHBoxLayout()

        copy_btn = QPushButton("Copy API Docs")
        copy_btn.clicked.connect(self._copy_api_docs)
        btn_layout.addWidget(copy_btn)

        copy_ai_btn = QPushButton("Copy for AI")
        copy_ai_btn.setToolTip("Copy comprehensive prompt for ChatGPT/Claude")
        copy_ai_btn.clicked.connect(self._copy_for_ai_prompt)
        btn_layout.addWidget(copy_ai_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Documentation
        docs = QTextEdit()
        docs.setReadOnly(True)
        docs.setHtml(self._get_api_docs_html())
        layout.addWidget(docs)

        return widget

    def _get_api_docs_html(self) -> str:
        """Get API documentation as HTML."""
        return '''
<h2>Pattern Script API</h2>

<h3>Script Header</h3>
<pre>
--[[
Pattern: PATTERN_NAME
DisplayName: Friendly Name With Spaces
Description: What it matches
Tier: 1-10 (1=rarest, 10=common)
Examples: ["12345678", "87654321"]
DataFile: optional_data.csv
--]]
</pre>
<p><b>DisplayName</b> is optional - if provided, it's shown in the GUI instead of the pattern name.</p>
<p><b style="color: #c9622c;">Examples</b> is required for the random preview generator to work. Without it, clicking "Generate Random" won't produce matching serials.</p>

<h3>Input Context (ctx)</h3>
<pre>
ctx.digits      -- "12345678" (8 numeric characters)
ctx.full_serial -- "A12345678B" (with prefix/suffix letters)
ctx.digit_list  -- {1,2,3,4,5,6,7,8} as integer array
ctx.metadata    -- {} additional detection data
ctx.data        -- External data (if DataFile specified)
ctx.data_by_key -- Key lookup dict (CSV only, keyed by first column)
</pre>

<h3>External Data Files (DataFile)</h3>
<p>Patterns can load CSV or JSON files for lookup tables:</p>
<pre>
-- CSV: loaded as list + key lookup
-- File: known_serials.csv
-- serial,description,value
-- 12345678,Ladder,$500

local entry = ctx.data_by_key[ctx.digits]
if entry then
    return {matched = true, message = entry.description}
end

-- JSON: loaded as-is (any structure)
-- File: dates.json
-- {"dates": {"07041776": {"name": "July 4th"}}}

local entry = ctx.data.dates[ctx.digits]
</pre>

<h3>Return Value</h3>
<pre>
return {
    matched = true,  -- or false
    highlights = {
        {positions = {0, 7}, color = "orange", label = "pair"},
    },
    connectors = {
        {from = 0, to = 7, color = "orange", style = "arc"},
    },
    group_boxes = {
        {from = 0, to = 2, color = "gold", thickness = 3},
    },
    message = "Optional description"
}
</pre>

<h3>Available Colors</h3>
<ul>
<li><b>purple</b> - Flipper-valid digits (0,1,6,8,9)</li>
<li><b>blue</b> - Binary patterns (0,1)</li>
<li><b>cyan</b> - Trinary/descending</li>
<li><b>orange</b> - Primary pairs (radar)</li>
<li><b>coral</b> - Secondary pairs</li>
<li><b>gold</b> - Quads/runs/known serials</li>
<li><b>salmon</b> - Tertiary pairs</li>
<li><b>magenta</b> - Repeater</li>
<li><b>yellow</b> - Solid/dominant/peaks</li>
<li><b>lime</b> - Ladder/ascending</li>
<li><b>teal</b> - Double pairs</li>
<li><b>red</b> - Errors/broken patterns</li>
<li><b>gray</b> - Neutral/other</li>
</ul>

<h3>Connector Styles</h3>
<ul>
<li><b>arc</b> - Curved line above digits</li>
<li><b>line</b> - Straight line</li>
<li><b>dashed</b> - Dashed line</li>
<li><b>bracket</b> - Bracket connector</li>
<li><b>arrow</b> - Arrow connector</li>
</ul>

<h3>Helper Functions</h3>
<pre>
-- Analysis
count_digits(s)           -- {["0"]=2, ["1"]=3, ...}
find_runs(s)              -- {{digit, start, length}, ...}
unique_count(s)           -- number of unique digits
digit_sum(s)              -- sum of all digits
most_common(s)            -- digit, count
get_unique_digits(s)      -- sorted unique digits as string

-- Pattern checks
is_ladder(s), is_ascending(s), is_descending(s)
is_palindrome(s)
is_repeater(s)            -- ABCDABCD
is_alternating(s)         -- XYXYXYXY
has_n_consecutive(s, n)   -- N identical in a row
all_flip_valid(s)         -- all digits are 0,1,6,8,9
flip_string(s)            -- 180-degree rotation

-- String utilities
only_digits(s, allowed)   -- s contains only allowed
starts_with(s, prefix), ends_with(s, suffix)
contains(s, substr)
is_bookended(s, n)        -- first N == last N

-- Visualization helpers
highlight(positions, color, label)
highlight_range(start, stop, color, label)
connector(from, to, color, style)
find_digit_positions(s, digit)
</pre>

<h3>Debug Logging</h3>
<pre>
log(value1, value2, ...)  -- Log values for debugging
</pre>
<p>Use <code>log()</code> to trace script execution during testing. Values are concatenated with spaces, tables are displayed as <code>{key=value, ...}</code>.</p>
<p>Logs appear in batch test results and are included in "Copy for AI Debug" output.</p>
<pre>
function match(ctx)
    log("digits:", ctx.digits)
    local count = unique_count(ctx.digits)
    log("unique count:", count)

    if count <= 2 then
        log("matched!")
        return {matched = true, message = "Binary"}
    end

    log("no match")
    return {matched = false}
end
</pre>

<h3>Example: Palindrome Pattern</h3>
<pre>
function match(ctx)
    if not is_palindrome(ctx.digits) then
        return {matched = false}
    end

    local colors = {"orange", "coral", "gold", "salmon"}
    local highlights = {}
    local connectors = {}

    for i = 0, 3 do
        local j = 7 - i
        table.insert(highlights, {positions = {i, j}, color = colors[i+1]})
        table.insert(connectors, {from = i, to = j, color = colors[i+1], style = "arc"})
    end

    return {matched = true, highlights = highlights, connectors = connectors}
end
</pre>
'''

    def _copy_api_docs(self):
        """Copy API docs to clipboard for pasting into AI chat."""
        docs = '''# Lua Pattern Script API for Dollar Bill Serial Numbers

## Script Header
```lua
--[[
Pattern: PATTERN_NAME
DisplayName: Friendly Name With Spaces
Description: What this pattern matches
Tier: 1-10 (1=rarest, 10=common)
Examples: ["12345678", "87654321"]
DataFile: optional_data.csv
--]]
```

**DisplayName** is optional - if provided, it's shown in the GUI instead of the pattern name (which must be uppercase with underscores for internal use).

**IMPORTANT: Examples is required** for the random preview generator to work. Without examples, clicking "Generate Random" won't produce matching serials.

## Input Context
The `ctx` table is available in every pattern script:
- ctx.digits: "12345678" (8 numeric characters)
- ctx.full_serial: "A12345678B" (with prefix/suffix letters)
- ctx.digit_list: {1,2,3,4,5,6,7,8} as integer array (1-indexed in Lua)
- ctx.metadata: {} additional detection metadata
- ctx.data: External data loaded from DataFile (if specified)
- ctx.data_by_key: Key lookup dict for CSV files (keyed by first column)

## External Data Files (DataFile header)
Patterns can declare a DataFile to load external CSV or JSON data:

**CSV files** - Loaded as list of row dicts + automatic key lookup:
```lua
-- DataFile: known_serials.csv
-- CSV format: serial,description,value
--             12345678,Perfect ladder,$500

local entry = ctx.data_by_key[ctx.digits]  -- O(1) lookup by first column
if entry then
    return {matched = true, message = entry.description .. " - " .. entry.value}
end

-- Or iterate all rows:
for _, row in ipairs(ctx.data) do
    if row.serial == ctx.digits then ...
end
```

**JSON files** - Loaded as-is (any structure):
```lua
-- DataFile: special_dates.json
-- JSON: {"dates": {"07041776": {"name": "July 4th", "significance": "Independence"}}}

if ctx.data and ctx.data.dates then
    local entry = ctx.data.dates[ctx.digits]
    if entry then
        return {matched = true, message = entry.name}
    end
end
```

**Path resolution:**
- Filename only (e.g., `data.csv`): same directory as .lua file
- `data/` prefix (e.g., `data/shared.csv`): patterns/data/ directory

## Return Value
The match function must return a table with:
- matched: boolean (required - true if pattern matches)
- highlights: list of {positions = {0, 7}, color = "orange", label = "optional"}
- connectors: list of {from = 0, to = 7, color = "orange", style = "arc"}
- group_boxes: list of {from = 0, to = 2, color = "gold", thickness = 3} (box around digit range)
- message: optional string describing the match

## Available Colors
purple, blue, cyan, orange, coral, gold, salmon, magenta, yellow, lime, teal, red, gray

## Connector Styles
arc, line, dashed, bracket, arrow

## Helper Functions

### Analysis
- count_digits(s): returns table {["0"]=count, ["1"]=count, ...}
- find_runs(s): finds consecutive runs, returns {{digit, start, length}, ...}
- unique_count(s): number of unique digits in string
- digit_sum(s): sum of all digits
- most_common(s): returns most_digit, count
- get_unique_digits(s): returns sorted unique digits as string

### Pattern Checks
- is_ladder(s), is_ascending(s), is_descending(s): ladder pattern checks
- find_ladder_of_length(s, min_len): find ladder of given minimum length
- find_longest_ladder(s): find the longest ladder in string
- is_palindrome(s): true if string equals its reverse
- is_broken_palindrome(s, max_mismatches): near-palindrome check
- is_repeater(s): true if ABCDABCD pattern
- is_super_repeater(s): true if ABABABAB pattern
- is_alternating(s): true if XYXYXYXY pattern
- has_n_consecutive(s, n): true if N identical digits in a row
- all_flip_valid(s): true if all digits are 0,1,6,8,9
- flip_string(s): returns 180-degree rotated version

### String Utilities
- only_digits(s, allowed): true if s contains only digits in allowed string
- starts_with(s, prefix), ends_with(s, suffix): prefix/suffix checks
- contains(s, substr): substring check
- is_bookended(s, n): true if first N digits == last N digits

### Pair/Group Detection
- find_pairs(s): find consecutive identical pairs
- find_consecutive_pairs(s): find pairs with positions
- has_four_consecutive_pairs(s): true if AABBCCDD pattern
- count_pairs(s): total number of pairs
- find_triples(s): find triple runs
- find_quads(s): find quad+ runs

### Visualization Helpers
- highlight(positions, color, label): build highlight entry
- highlight_range(start, stop, color, label): highlight range of positions
- connector(from, to, color, style): build connector entry
- find_digit_positions(s, digit): get all positions of a specific digit

## Debug Logging
Use log() to trace script execution during testing:
```lua
log(value1, value2, ...)  -- values are space-separated, tables shown as {k=v, ...}
```

Example:
```lua
function match(ctx)
    log("digits:", ctx.digits)
    local count = unique_count(ctx.digits)
    log("unique count:", count)

    if count <= 2 then
        log("matched!")
        return {matched = true, message = "Binary"}
    end

    log("no match")
    return {matched = false}
end
```

Logs appear in batch test results and are included in "Copy for AI Debug" output.

## Example: Palindrome Pattern
```lua
--[[
Pattern: MY_RADAR
Description: Serial reads same forwards and backwards
Tier: 3
Examples: ["12344321", "45677654"]
--]]

function match(ctx)
    if not is_palindrome(ctx.digits) then
        return {matched = false}
    end

    local colors = {"orange", "coral", "gold", "salmon"}
    local highlights = {}
    local connectors = {}

    for i = 0, 3 do
        local j = 7 - i
        table.insert(highlights, {positions = {i, j}, color = colors[i+1]})
        table.insert(connectors, {from = i, to = j, color = colors[i+1], style = "arc"})
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = connectors,
        message = "Palindrome serial number"
    }
end
```

## Example: Pattern with External Data
```lua
--[[
Pattern: KNOWN_SERIALS
Description: Match against database of known collectible serials
Tier: 1
DataFile: known_serials.csv
--]]

function match(ctx)
    if not ctx.data_by_key then
        return {matched = false}
    end

    local entry = ctx.data_by_key[ctx.digits]
    if entry then
        return {
            matched = true,
            highlights = {highlight_range(0, 7, "gold", "Known serial")},
            message = entry.description .. " - " .. entry.value
        }
    end

    return {matched = false}
end
```
'''
        clipboard = QApplication.clipboard()
        clipboard.setText(docs)
        QMessageBox.information(self, "Copied", "API documentation copied to clipboard!")

    def _copy_for_ai_prompt(self):
        """Copy comprehensive AI prompt with docs, helpers, and template."""
        # Load helper function reference
        helpers_content = self._load_helpers_reference()

        # Build the full AI prompt
        prompt = self._build_ai_prompt_template(helpers_content)

        clipboard = QApplication.clipboard()
        clipboard.setText(prompt)
        QMessageBox.information(self, "Copied", "AI prompt copied to clipboard!\n\nPaste into ChatGPT/Claude to get help writing your pattern.")

    def _load_helpers_reference(self) -> str:
        """Read helpers.lua and extract function signatures with docs."""
        helpers_path = Path(__file__).parent.parent / "patterns" / "lib" / "helpers.lua"
        if not helpers_path.exists():
            return "(helpers.lua not found)"

        try:
            content = helpers_path.read_text(encoding='utf-8')
        except Exception as e:
            return f"(Error reading helpers.lua: {e})"

        # Parse functions with their comments
        functions = []
        lines = content.split('\n')
        current_comment = []

        for line in lines:
            stripped = line.strip()

            # Collect comments
            if stripped.startswith('--') and not stripped.startswith('--[['):
                # Single-line comment
                comment_text = stripped[2:].strip()
                if comment_text:
                    current_comment.append(comment_text)
            elif stripped.startswith('function ') and '(' in stripped:
                # Function definition
                match = re.match(r'function\s+(\w+)\s*\(([^)]*)\)', stripped)
                if match:
                    func_name = match.group(1)
                    params = match.group(2)
                    doc = ' '.join(current_comment) if current_comment else ''
                    functions.append(f"- {func_name}({params}): {doc}" if doc else f"- {func_name}({params})")
                current_comment = []
            elif stripped and not stripped.startswith('--'):
                # Non-comment, non-function line - reset comment accumulator
                current_comment = []

        return '\n'.join(functions)

    def _build_ai_prompt_template(self, helpers_content: str) -> str:
        """Construct the full AI prompt for pattern creation."""
        # Get the plain text API docs
        api_docs = self._get_api_docs_plain_text()

        prompt = f'''You are helping write Lua patterns for dollar bill serial number classification. These patterns analyze 8-digit serial numbers and return match results with optional visual highlighting.

## API Documentation

{api_docs}

## Helper Functions Reference (from helpers.lua)

The following helper functions are automatically available in all pattern scripts:

{helpers_content}

## Your Task

Create a Lua pattern script for the following:

**Pattern Name:** [PATTERN_NAME]

**Description:** [DESCRIPTION]

**Should Match (examples):**
[SHOULD_MATCH_EXAMPLES - one per line]

**Should NOT Match (examples):**
[SHOULD_NOT_MATCH_EXAMPLES - one per line]

## Validation Rules & Common Pitfalls

1. **Return structure:** Always return a table with `matched = true/false`. Include `highlights`, `connectors`, and/or `message` when matched.

2. **Position indexing:** All positions are 0-indexed (0-7 for 8 digits), but Lua strings are 1-indexed. Use `ctx.digits:sub(i+1, i+1)` to get the character at position `i`.

3. **ctx.digit_list:** This is a 1-indexed Lua array of integers: `{{1,2,3,4,5,6,7,8}}`. Use `ctx.digit_list[1]` for the first digit.

4. **Use helper functions:** Don't reinvent - use `is_palindrome()`, `find_runs()`, `count_digits()`, etc.

5. **Highlights vs group_boxes:** Use `highlights` for individual digit positions, `group_boxes` for spanning a range of consecutive digits.

6. **Test edge cases:** Consider what happens with all-same digits (11111111), ascending (12345678), palindromes, etc.

7. **Debug with log():** Use `log("message", value)` to trace execution. Logs appear in batch test results and "Copy for AI Debug" output.

## Response Format

Provide the complete Lua script including the header comment block with Pattern, Description, Tier, and Examples fields.
'''
        return prompt

    def _get_api_docs_plain_text(self) -> str:
        """Get API documentation as plain text for AI prompts."""
        # Reuse the content from _copy_api_docs but return it instead of copying
        return '''# Lua Pattern Script API for Dollar Bill Serial Numbers

## Script Header
```lua
--[[
Pattern: PATTERN_NAME
DisplayName: Friendly Name With Spaces
Description: What this pattern matches
Tier: 1-10 (1=rarest, 10=common)
Examples: ["12345678", "87654321"]
DataFile: optional_data.csv
--]]
```

**DisplayName** is optional - if provided, it's shown in the GUI instead of the pattern name.

**IMPORTANT: Examples is required** for the random preview generator to work. Without examples, clicking "Generate Random" won't produce matching serials.

## Input Context
The `ctx` table is available in every pattern script:
- ctx.digits: "12345678" (8 numeric characters)
- ctx.full_serial: "A12345678B" (with prefix/suffix letters)
- ctx.digit_list: {1,2,3,4,5,6,7,8} as integer array (1-indexed in Lua)
- ctx.metadata: {} additional detection metadata
- ctx.data: External data loaded from DataFile (if specified)
- ctx.data_by_key: Key lookup dict for CSV files (keyed by first column)

## Return Value
The match function must return a table with:
- matched: boolean (required - true if pattern matches)
- highlights: list of {positions = {0, 7}, color = "orange", label = "optional"}
- connectors: list of {from = 0, to = 7, color = "orange", style = "arc"}
- group_boxes: list of {from = 0, to = 2, color = "gold", thickness = 3}
- message: optional string describing the match

## Available Colors
purple, blue, cyan, orange, coral, gold, salmon, magenta, yellow, lime, teal, red, gray

## Connector Styles
arc, line, dashed, bracket, arrow'''

    def _parse_serials(self, text: str) -> list:
        """Parse comma-delimited text into list of 8-digit serials."""
        serials = []
        # Split by comma, then process each entry
        for entry in text.strip().split(','):
            entry = entry.strip()
            if not entry:
                continue
            # Extract digits only
            digits = ''.join(c for c in entry if c.isdigit())
            if len(digits) == 8:
                serials.append(digits)
            elif len(digits) == 10 and entry[0].isalpha() and entry[-1].isalpha():
                # Full serial like A12345678B
                serials.append(digits)
        return serials

    def _test_single_case(self, script: str, serial: str, expected: bool, debug: bool = False) -> dict:
        """Test one serial against the script, return result dict."""
        result = {
            'serial': serial,
            'expected': expected,
            'actual': False,
            'passed': False,
            'error': '',
            'message': '',
            'debug_log': []
        }

        if not self.engine:
            result['error'] = 'No pattern engine available'
            return result

        try:
            # Add prefix/suffix if just digits
            full_serial = f"A{serial}B" if len(serial) == 8 and serial.isdigit() else serial
            test_result = self.engine.test_script(script, full_serial, debug=debug)

            if test_result.success:
                result['actual'] = test_result.matched
                result['message'] = test_result.message or ''
                result['debug_log'] = test_result.debug_log or []
            else:
                result['error'] = test_result.error or 'Unknown error'
                result['actual'] = False
                result['debug_log'] = test_result.debug_log or []
        except Exception as e:
            result['error'] = str(e)
            result['actual'] = False

        result['passed'] = (result['actual'] == expected) and not result['error']
        return result

    def _run_batch_tests(self):
        """Execute all test cases and display results."""
        if not self.engine:
            self.test_results.setHtml("<span style='color: red;'>No pattern engine available</span>")
            return

        script = self.script_edit.toPlainText()
        self._last_script = script

        # Parse test cases
        should_match = self._parse_serials(self.should_match_edit.text())
        should_not_match = self._parse_serials(self.should_not_match_edit.text())

        if not should_match and not should_not_match:
            self.test_results.setHtml("<span style='color: gray;'>Enter test serials above (one per line)</span>")
            self.copy_debug_btn.setEnabled(False)
            return

        results = []
        passed = 0
        failed = 0

        # Test should-match cases (with debug enabled)
        for serial in should_match:
            result = self._test_single_case(script, serial, expected=True, debug=True)
            results.append(result)
            if result['passed']:
                passed += 1
            else:
                failed += 1

        # Test should-not-match cases (with debug enabled)
        for serial in should_not_match:
            result = self._test_single_case(script, serial, expected=False, debug=True)
            results.append(result)
            if result['passed']:
                passed += 1
            else:
                failed += 1

        self._last_batch_results = results
        self._display_batch_results(results, passed, failed)

        # Enable debug button if there are failures
        self.copy_debug_btn.setEnabled(failed > 0)

    def _display_batch_results(self, results: list, passed: int, failed: int):
        """Render HTML results for batch tests."""
        total = passed + failed

        if failed == 0:
            summary_color = "green"
            summary = f"All {total} tests passed!"
        else:
            summary_color = "red"
            summary = f"{passed}/{total} passed, {failed} failed"

        html = f"<b style='color: {summary_color};'>{summary}</b><br><br>"

        for result in results:
            if result['passed']:
                icon = "✓"
                color = "green"
            else:
                icon = "✗"
                color = "red"

            expected_str = "should match" if result['expected'] else "should NOT match"
            actual_str = "matched" if result['actual'] else "no match"

            html += f"<span style='color: {color};'>{icon}</span> "
            html += f"<code>{result['serial']}</code> - {expected_str}, {actual_str}"

            if result['error']:
                html += f" <span style='color: orange;'>(Error: {result['error']})</span>"
            elif result['message'] and result['actual']:
                html += f" <span style='color: gray;'>({result['message']})</span>"

            html += "<br>"

            # Show debug log entries if present
            debug_log = result.get('debug_log', [])
            if debug_log:
                html += "<div style='margin-left: 20px; margin-bottom: 8px;'>"
                for entry in debug_log:
                    # Escape HTML entities in log entry
                    safe_entry = str(entry).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    html += f"<code style='color: #888; font-size: 0.9em;'>log: {safe_entry}</code><br>"
                html += "</div>"

        self.test_results.setHtml(html)

    def _export_test_cases(self):
        """Copy test cases formatted for AI prompts."""
        should_match = self._parse_serials(self.should_match_edit.text())
        should_not_match = self._parse_serials(self.should_not_match_edit.text())

        if not should_match and not should_not_match:
            QMessageBox.information(self, "No Test Cases", "Enter test serials first.")
            return

        export = "## Test Cases\n\n"

        if should_match:
            export += "**Should Match:**\n"
            for serial in should_match:
                export += f"- {serial}\n"
            export += "\n"

        if should_not_match:
            export += "**Should NOT Match:**\n"
            for serial in should_not_match:
                export += f"- {serial}\n"

        clipboard = QApplication.clipboard()
        clipboard.setText(export)
        QMessageBox.information(self, "Copied", "Test cases copied to clipboard!")

    def _copy_debug_info(self):
        """Copy failing test context for AI debugging assistance."""
        if not self._last_batch_results:
            return

        failures = [r for r in self._last_batch_results if not r['passed']]
        if not failures:
            QMessageBox.information(self, "No Failures", "All tests passed!")
            return

        prompt = self._build_debug_prompt(failures)

        clipboard = QApplication.clipboard()
        clipboard.setText(prompt)
        QMessageBox.information(self, "Copied", f"Debug info for {len(failures)} failing test(s) copied to clipboard!")

    def _build_debug_prompt(self, failures: list) -> str:
        """Construct debug prompt with script + failures + debug logs."""
        # Build context example using the first failing serial
        sample_serial = failures[0]['serial'] if failures else "12345678"
        digit_list_str = ', '.join(sample_serial)

        prompt = f'''I need help debugging a Lua pattern for dollar bill serial number classification.

## Context (ctx) Contents

For serial "{sample_serial}":
- `ctx.digits` = "{sample_serial}" (8-digit string)
- `ctx.full_serial` = "A{sample_serial}B" (with prefix/suffix)
- `ctx.digit_list` = {{{digit_list_str}}} (1-indexed array of integers)
- `ctx.metadata` = {{}} (may contain series_year, front_plate, back_plate)

## Current Script

```lua
{self._last_script}
```

## Failing Tests

'''
        has_debug_logs = False
        for f in failures:
            expected_str = "should match" if f['expected'] else "should NOT match"
            actual_str = "matched" if f['actual'] else "did not match"
            prompt += f"### Serial: {f['serial']}\n"
            prompt += f"- Expected: {expected_str}\n"
            prompt += f"- Actual: {actual_str}\n"
            if f['error']:
                prompt += f"- Error: {f['error']}\n"

            # Include debug log if present
            debug_log = f.get('debug_log', [])
            if debug_log:
                has_debug_logs = True
                prompt += "- Debug log:\n"
                for entry in debug_log:
                    prompt += f"  - `{entry}`\n"
            prompt += "\n"

        prompt += '''## Request

Please analyze why these tests are failing and provide a corrected version of the script. Explain what was wrong and how your fix addresses it.
'''

        # Add suggestion to use log() if no debug logs were found
        if not has_debug_logs:
            prompt += '''
**Tip:** The script has no `log()` calls. Consider adding debug logging to trace execution:
```lua
log("digits:", ctx.digits)
log("checking condition:", some_value)
```
'''
        return prompt

    def _update_hint(self):
        """Update the hint based on rule type."""
        rule = self.rule_type.currentText()
        hints = {
            "contains": "Matches if serial contains this exact value anywhere.\nSingle value only. Use 'regex' with | for multiple values.\nExamples: '0704' matches July 4, '1990' matches birth year",
            "starts_with": "Matches if serial starts with this exact value.\nSingle value only. Example: '000' matches low serial numbers",
            "ends_with": "Matches if serial ends with this exact value.\nSingle value only. Example: '0000' matches round numbers",
            "regex": "Regular expression pattern. Use | for OR logic.\nExamples: '0704|1225' matches July 4 or Christmas,\n'(\\d)\\1{3}' matches 4 repeated digits",
            "baseline_variance_min": "Matches if baseline variance >= this value.\nUsed for gas pump detection.",
            "baseline_variance_max": "Matches if baseline variance <= this value."
        }
        self.hint_label.setText(hints.get(rule, ""))

    def _validate_script(self):
        """Validate the Lua script syntax."""
        if not self.engine:
            return

        script = self.script_edit.toPlainText()
        valid, error = self.engine.validate_script(script)

        if valid:
            self.script_status.setText("Syntax OK")
            self.script_status.setStyleSheet("color: green;")
        else:
            self.script_status.setText(f"Syntax Error: {error}")
            self.script_status.setStyleSheet("color: red;")

    def _format_script(self):
        """Auto-format the Lua script (basic indentation)."""
        script = self.script_edit.toPlainText()
        lines = script.split('\n')
        formatted = []
        indent = 0

        for line in lines:
            stripped = line.strip()

            # Decrease indent for end, else, elseif, until
            if stripped.startswith(('end', 'else', 'elseif', 'until', '}')):
                indent = max(0, indent - 1)

            formatted.append('    ' * indent + stripped)

            # Increase indent for function, if, for, while, repeat, do
            if stripped.endswith(('then', 'do', 'function', 'repeat', '{')):
                indent += 1
            elif stripped.startswith('function') and not stripped.endswith('end'):
                indent += 1

        self.script_edit.setPlainText('\n'.join(formatted))

    def _insert_template(self, template_name: str):
        """Insert a code template."""
        templates = {
            "Basic Match": '''function match(ctx)
    -- Basic pattern matching template
    if ctx.digits == "12345678" then
        return {
            matched = true,
            highlights = {{positions = {0,1,2,3,4,5,6,7}, color = "orange"}},
            message = "Matched!"
        }
    end
    return {matched = false}
end
''',
            "Palindrome Check": '''function match(ctx)
    -- Check if serial is a palindrome
    local rev = string.reverse(ctx.digits)
    if ctx.digits ~= rev then
        return {matched = false}
    end

    local colors = {"orange", "coral", "gold", "salmon"}
    local highlights = {}
    local connectors = {}

    for i = 0, 3 do
        local j = 7 - i
        table.insert(highlights, {positions = {i, j}, color = colors[i+1]})
        table.insert(connectors, {from = i, to = j, color = colors[i+1], style = "arc"})
    end

    return {matched = true, highlights = highlights, connectors = connectors}
end
''',
            "Run Detection": '''function match(ctx)
    -- Find runs of consecutive same digits
    local digits = ctx.digits
    local highlights = {}
    local has_run = false

    local i = 1
    while i <= #digits do
        local d = digits:sub(i, i)
        local run_start = i - 1  -- 0-indexed
        local run_len = 1

        while i + run_len <= #digits and digits:sub(i + run_len, i + run_len) == d do
            run_len = run_len + 1
        end

        if run_len >= 3 then
            has_run = true
            local positions = {}
            for p = run_start, run_start + run_len - 1 do
                table.insert(positions, p)
            end
            table.insert(highlights, {positions = positions, color = "gold", label = "run"})
        end

        i = i + run_len
    end

    if has_run then
        return {matched = true, highlights = highlights}
    end
    return {matched = false}
end
''',
            "Digit Count": '''function match(ctx)
    -- Match based on digit frequency
    local counts = {}
    for i = 1, #ctx.digits do
        local d = ctx.digits:sub(i, i)
        counts[d] = (counts[d] or 0) + 1
    end

    -- Check for 6+ of same digit
    for d, c in pairs(counts) do
        if c >= 6 then
            local highlights = {}
            for i = 0, 7 do
                if ctx.digits:sub(i+1, i+1) == d then
                    table.insert(highlights, {positions = {i}, color = "gold"})
                end
            end
            return {
                matched = true,
                highlights = highlights,
                message = c .. " x " .. d
            }
        end
    end

    return {matched = false}
end
''',
        }

        if template_name in templates:
            self.script_edit.setPlainText(templates[template_name])

    def _run_live_test(self):
        """Run a live test of the pattern."""
        if not self.engine:
            return

        serial = self.test_serial_edit.text().strip()
        if not serial:
            return

        # Add prefix/suffix if missing
        digits = ''.join(c for c in serial if c.isdigit())
        if len(serial) == 8 and serial.isdigit():
            serial = f"A{serial}B"

        self.digit_preview.set_serial(serial)

        # Check if we should run Lua script test (Script tab or Test tab with Lua pattern)
        current_tab = self.tab_widget.currentIndex()
        is_lua_mode = current_tab == 1 or (current_tab == 2 and self.is_lua_pattern)

        if is_lua_mode:
            # Lua script test
            script = self.script_edit.toPlainText()
            result = self.engine.test_script(script, serial)

            if result.success:
                self.digit_preview.set_highlights(result.highlights, result.connectors)
                # Also set group_boxes if available
                if hasattr(result, 'group_boxes') and result.group_boxes:
                    self.digit_preview.set_group_boxes(result.group_boxes)

                if result.matched:
                    self.test_results.setText(
                        f"MATCHED!\n\n"
                        f"Message: {result.message or '(none)'}\n"
                        f"Highlights: {len(result.highlights)}\n"
                        f"Connectors: {len(result.connectors)}\n"
                        f"Execution time: {result.execution_time_ms:.2f}ms"
                    )
                    self.test_results.setStyleSheet("color: green;")
                else:
                    self.test_results.setText("No match")
                    self.test_results.setStyleSheet("color: gray;")
                    self.digit_preview.set_highlights([], [])
            else:
                self.test_results.setText(f"Error: {result.error}")
                self.test_results.setStyleSheet("color: red;")
                self.digit_preview.set_highlights([], [])
        elif current_tab == 0:
            # Simple rule test (only on Simple Rule tab)
            self.digit_preview.set_highlights([], [])
            rule_type = self.rule_type.currentText()
            value = self.value_edit.text().strip()

            if not value:
                self.test_results.setText("Enter a value to test")
                return

            # Simple rule check
            matched = False
            if rule_type == "contains":
                matched = value in digits
            elif rule_type == "starts_with":
                matched = digits.startswith(value)
            elif rule_type == "ends_with":
                matched = digits.endswith(value)
            elif rule_type == "regex":
                import re
                matched = bool(re.search(value, digits))

            if matched:
                self.test_results.setText(f"MATCHED!\n\nRule: {rule_type}\nValue: {value}")
                self.test_results.setStyleSheet("color: green;")
            else:
                self.test_results.setText(f"No match\n\nRule: {rule_type}\nValue: {value}")
                self.test_results.setStyleSheet("color: gray;")

    def _auto_generate_display_name(self):
        """Auto-generate a friendly display name from the pattern name."""
        # Only auto-generate if display name is empty or was auto-generated
        current_display = self.display_name_edit.text()
        name = self.name_edit.text().strip()

        # Generate friendly name: LOW_RUN_6M -> Low Run 6M
        if name:
            words = name.replace('_', ' ').split()
            friendly_words = []
            for word in words:
                # Keep numbers/acronyms as-is but title-case regular words
                if word.isdigit() or (len(word) <= 3 and any(c.isdigit() for c in word)):
                    friendly_words.append(word.upper())
                else:
                    friendly_words.append(word.capitalize())
            generated = ' '.join(friendly_words)

            # Only update if display name is empty or matches previous auto-generation
            if not current_display or current_display == getattr(self, '_last_auto_display', ''):
                self.display_name_edit.setText(generated)
                self._last_auto_display = generated

    def _load_existing(self):
        """Load existing pattern data."""
        self.name_edit.setText(self.original_name)

        # Load display name - either from defn or auto-generate from name
        display_name = self.defn.get('display_name', '')
        if display_name:
            self.display_name_edit.setText(display_name)
        else:
            # Auto-generate from pattern name
            self._auto_generate_display_name()

        self.desc_edit.setText(self.defn.get('description', ''))
        self.tier_spin.setValue(self.defn.get('tier', 5))

        if self.original_script:
            # Lua script pattern
            if HAS_V3_ENGINE:
                self.tab_widget.setCurrentIndex(1)  # Script tab
                self.script_edit.setPlainText(self.original_script)
        else:
            # Simple rule pattern
            rules = self.defn.get('rules', {})
            for rule_type in ['contains', 'starts_with', 'ends_with', 'regex',
                              'baseline_variance_min', 'baseline_variance_max']:
                if rule_type in rules:
                    self.rule_type.setCurrentText(rule_type)
                    self.value_edit.setText(str(rules[rule_type]))
                    break

    def _validate_and_accept(self):
        """Validate input and accept."""
        name = self.name_edit.text().strip().upper().replace(' ', '_')
        if not name:
            QMessageBox.warning(self, "Validation Error", "Please enter a pattern name.")
            return

        # Check which mode we're in based on:
        # 1. If editing an existing Lua pattern (is_lua_pattern flag) → Lua mode
        # 2. If user is on Pattern Wizard tab (index 1) → Wizard mode
        # 3. If user is on AI Generate tab (index 2) → AI mode (use generated code)
        # 4. If user is on Lua Script tab (index 3) → Lua mode (explicit choice)
        # 5. If script has header block (--[[) → Lua mode (user customized script)
        # 6. Otherwise → Simple Rule mode (default for new patterns)
        script = self.script_edit.toPlainText().strip() if HAS_V3_ENGINE else ""
        current_tab = self.tab_widget.currentIndex() if HAS_V3_ENGINE else 0
        has_custom_script = script.startswith('--[[')  # Default template doesn't have header

        # Check for wizard mode (tab index 1)
        use_wizard_mode = HAS_V3_ENGINE and current_tab == 1

        # Check for AI mode (tab index 2)
        use_ai_mode = HAS_V3_ENGINE and current_tab == 2

        # Use Lua mode if: editing a Lua pattern, on Lua Script tab (index 3), or script has header
        use_lua_mode = self.is_lua_pattern or current_tab == 3 or has_custom_script

        if use_wizard_mode:
            # Wizard mode - validate that we have a valid recipe and parameters
            recipe = self._get_current_recipe()
            if not recipe:
                QMessageBox.warning(self, "Validation Error", "Please select a recipe type.")
                return

            params = self._get_wizard_params()

            # For digit set, ensure at least 1 digit is selected
            if recipe.name == "digit_set":
                digits = params.get("digits", [])
                if not digits:
                    QMessageBox.warning(self, "Validation Error", "Please select at least one digit.")
                    return

            # Generate and validate the script
            try:
                color = self.wizard_color_combo.currentText()
                description = self.desc_edit.text().strip() or recipe.description
                tier = self.tier_spin.value()
                lua_code = recipe.generate_lua(params, name, description, tier, color)

                valid, error = self.engine.validate_script(lua_code)
                if not valid:
                    QMessageBox.warning(self, "Script Error", f"Generated Lua has syntax error:\n{error}")
                    return

                # Store the generated script for get_pattern()
                self._wizard_generated_script = lua_code
                self._wizard_examples = recipe.generate_examples(params)
                self.is_lua_pattern = True
                self._is_wizard_pattern = True
            except Exception as e:
                QMessageBox.warning(self, "Generation Error", f"Failed to generate pattern:\n{str(e)}")
                return

        elif use_ai_mode:
            # AI mode - use the generated code if available
            if not hasattr(self, '_ai_generated_code') or not self._ai_generated_code:
                QMessageBox.warning(
                    self,
                    "No Generated Code",
                    "Please generate a pattern using the AI first, or switch to another tab."
                )
                return

            # Validate the generated script
            valid, error = self.engine.validate_script(self._ai_generated_code)
            if not valid:
                QMessageBox.warning(self, "Script Error", f"Generated Lua has syntax error:\n{error}")
                return

            # Store for get_pattern()
            self._wizard_generated_script = self._ai_generated_code
            self._wizard_examples = []  # AI-generated code should have Examples in header
            self.is_lua_pattern = True
            self._is_wizard_pattern = True  # Reuse wizard pattern flow

        elif HAS_V3_ENGINE and use_lua_mode:
            # Script mode - validate syntax
            valid, error = self.engine.validate_script(script)
            if not valid:
                QMessageBox.warning(self, "Script Error", f"Lua syntax error:\n{error}")
                return
            self.is_lua_pattern = True
            self._is_wizard_pattern = False

            # Check for missing Examples in header
            if 'Examples:' not in script and 'Examples :' not in script:
                reply = QMessageBox.warning(
                    self,
                    "Missing Examples",
                    "Your pattern is missing the Examples field in the header.\n\n"
                    "Without examples, the random serial generator won't work in the preview.\n\n"
                    "Add a line like:\nExamples: [\"12345678\", \"87654321\"]\n\n"
                    "Save anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
        else:
            # Simple rule mode
            value = self.value_edit.text().strip()
            if not value:
                QMessageBox.warning(self, "Validation Error", "Please enter a value to match.")
                return
            self.is_lua_pattern = False
            self._is_wizard_pattern = False

        self.accept()

    def _generate_examples_for_rule(self, rule_type: str, value: str) -> list:
        """Generate example serials that match the given simple rule."""
        examples = []

        if rule_type == "contains":
            # Place the value at different positions
            val_len = len(value)
            if val_len <= 8:
                # At start
                examples.append(value + "0" * (8 - val_len))
                # In middle (if room)
                if val_len <= 6:
                    pad = (8 - val_len) // 2
                    examples.append("0" * pad + value + "0" * (8 - val_len - pad))
                # At end
                examples.append("0" * (8 - val_len) + value)

        elif rule_type == "starts_with":
            val_len = len(value)
            if val_len <= 8:
                examples.append(value + "0" * (8 - val_len))
                examples.append(value + "1" * (8 - val_len))
                examples.append(value + "2" * (8 - val_len))

        elif rule_type == "ends_with":
            val_len = len(value)
            if val_len <= 8:
                examples.append("0" * (8 - val_len) + value)
                examples.append("1" * (8 - val_len) + value)
                examples.append("2" * (8 - val_len) + value)

        elif rule_type == "regex":
            # For regex, we can't easily generate examples
            # User will need to add them manually or test will show warning
            pass

        # Filter to valid 8-digit examples
        examples = [ex for ex in examples if len(ex) == 8 and ex.isdigit()]
        return examples[:3]  # Return up to 3 examples

    def _generate_lua_from_rule(self, rule_type: str, value: str, description: str) -> str:
        """Generate a Lua script from a simple rule."""

        if rule_type == "contains":
            # Find positions where value appears for highlighting
            highlight_code = f'''    local pos = ctx.digits:find("{value}", 1, true)
    if pos then
        local positions = {{}}
        for i = 0, {len(value) - 1} do
            table.insert(positions, pos - 1 + i)
        end
        return {{
            matched = true,
            highlights = {{{{positions = positions, color = "orange"}}}},
            message = "{description}"
        }}
    end'''
            match_logic = highlight_code

        elif rule_type == "starts_with":
            positions = list(range(len(value)))
            match_logic = f'''    if starts_with(ctx.digits, "{value}") then
        return {{
            matched = true,
            highlights = {{{{positions = {{{", ".join(map(str, positions))}}}, color = "orange"}}}},
            message = "{description}"
        }}
    end'''

        elif rule_type == "ends_with":
            positions = list(range(8 - len(value), 8))
            match_logic = f'''    if ends_with(ctx.digits, "{value}") then
        return {{
            matched = true,
            highlights = {{{{positions = {{{", ".join(map(str, positions))}}}, color = "orange"}}}},
            message = "{description}"
        }}
    end'''

        elif rule_type == "regex":
            # Lua patterns are different from regex, but we can try basic conversion
            # For simple patterns, just use Lua's string.match
            match_logic = f'''    if ctx.digits:match("{value}") then
        return {{
            matched = true,
            message = "{description}"
        }}
    end'''

        else:
            # Fallback for unknown rule types
            match_logic = f'''    -- Rule type: {rule_type}, value: {value}
    return {{matched = false}}'''

        script = f'''function match(ctx)
{match_logic}
    return {{matched = false}}
end
'''
        return script

    def get_pattern(self) -> tuple:
        """Return the pattern name and definition."""
        name = self.name_edit.text().strip().upper().replace(' ', '_')
        tier = self.tier_spin.value()
        description = self.desc_edit.text().strip() or f"Custom pattern: {name}"
        display_name = self.display_name_edit.text().strip()

        # Check for wizard-generated pattern
        if getattr(self, '_is_wizard_pattern', False) and hasattr(self, '_wizard_generated_script'):
            script = self._wizard_generated_script
            examples = getattr(self, '_wizard_examples', [])
            return name, {
                'description': description,
                'display_name': display_name,
                'tier': tier,
                'script': script,
                'examples': examples,
                'source': 'lua'
            }
        elif self.is_lua_pattern and HAS_V3_ENGINE:
            # Return script from editor
            script = self.script_edit.toPlainText()
            return name, {
                'description': description,
                'display_name': display_name,
                'tier': tier,
                'script': script,
                'source': 'lua'
            }
        else:
            # Convert simple rule to Lua script
            rule_type = self.rule_type.currentText()
            value = self.value_edit.text().strip()

            # Generate examples and Lua script
            examples = self._generate_examples_for_rule(rule_type, value)
            lua_body = self._generate_lua_from_rule(rule_type, value, description)

            # Build full script with header
            import json
            header_lines = [
                '--[[',
                f'Pattern: {name}',
            ]
            if display_name:
                header_lines.append(f'DisplayName: {display_name}')
            header_lines.append(f'Description: {description}')
            header_lines.append(f'Tier: {tier}')
            if examples:
                header_lines.append(f'Examples: {json.dumps(examples)}')
            header_lines.append('--]]')
            header_lines.append('')

            script = '\n'.join(header_lines) + lua_body

            return name, {
                'description': description,
                'display_name': display_name,
                'tier': tier,
                'script': script,
                'examples': examples,
                'source': 'lua'
            }


# Test dialog standalone
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dialog = PatternDialog()
    dialog.exec()
