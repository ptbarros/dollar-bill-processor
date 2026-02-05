"""
Pattern Dialog - Manage pattern enable/disable and testing.

Supports both YAML-based simple rules and Lua script patterns.
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QGroupBox, QLineEdit, QPushButton, QDialogButtonBox, QLabel,
    QTextEdit, QSplitter, QHeaderView, QCheckBox, QListWidget,
    QListWidgetItem, QFormLayout, QComboBox, QMessageBox, QColorDialog,
    QTabWidget, QWidget, QSpinBox, QFrame, QApplication, QPlainTextEdit,
    QInputDialog
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat, QFontMetrics

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pattern_engine_v3 import PatternEngineV3 as PatternEngine
HAS_V3_ENGINE = True

from settings_manager import get_settings

# Try to import QScintilla for better code editing
try:
    from PyQt5.Qsci import QsciScintilla, QsciLexerLua
    HAS_QSCINTILLA = True
except ImportError:
    HAS_QSCINTILLA = False


class PatternDialog(QDialog):
    """Dialog for managing patterns and testing serials."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = PatternEngine()
        self.settings = get_settings()

        self.setWindowTitle("Pattern Manager")
        self.setMinimumSize(900, 600)
        self._setup_ui()
        self._load_patterns()

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
        self.pattern_tree.setSortingEnabled(True)

        header = self.pattern_tree.header()
        header.setSectionsMovable(False)  # Don't allow reordering columns
        header.setStretchLastSection(True)  # Last column fills remaining space
        header.setSortIndicatorShown(True)
        # All columns interactive (draggable) except last which stretches
        header.setSectionResizeMode(0, QHeaderView.Interactive)  # Pattern
        header.setSectionResizeMode(1, QHeaderView.Interactive)  # Tier
        header.setSectionResizeMode(2, QHeaderView.Interactive)  # Enabled
        header.setSectionResizeMode(3, QHeaderView.Interactive)  # Color
        header.setSectionResizeMode(4, QHeaderView.Stretch)      # Catalog

        # Restore saved column widths or use defaults
        self._restore_column_widths()

        # Save column widths when changed
        header.sectionResized.connect(self._save_column_widths)

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
        self.lua_script_layout.addStretch()
        details_layout.addLayout(self.lua_script_layout)

        # Initially hidden
        self.lua_script_label.hide()
        self.view_script_btn.hide()
        self._current_lua_pattern = None
        self._current_lua_editable = False  # True if pattern is not from 'core'

        right_layout.addWidget(details_group)

        # Serial tester
        test_group = QGroupBox("Serial Tester")
        test_layout = QVBoxLayout(test_group)

        test_input_layout = QHBoxLayout()
        self.test_edit = QLineEdit()
        self.test_edit.setPlaceholderText("Enter serial number (e.g., A12345678B)")
        self.test_edit.returnPressed.connect(self._test_serial)
        test_input_layout.addWidget(self.test_edit)

        test_btn = QPushButton("Test")
        test_btn.clicked.connect(self._test_serial)
        test_input_layout.addWidget(test_btn)

        test_layout.addLayout(test_input_layout)

        self.test_results = QTextEdit()
        self.test_results.setReadOnly(True)
        self.test_results.setMaximumHeight(150)
        test_layout.addWidget(self.test_results)

        right_layout.addWidget(test_group)

        # Quick test examples
        examples_group = QGroupBox("Quick Test Examples")
        examples_layout = QHBoxLayout(examples_group)

        examples = [
            ("Radar", "A12344321B"),
            ("Repeater", "A12341234B"),
            ("Binary", "A10101010B"),
            ("Ladder", "A12345678B"),
            ("Star", "A12345678*"),
        ]

        for label, serial in examples:
            btn = QPushButton(label)
            btn.setToolTip(serial)
            btn.clicked.connect(lambda checked, s=serial: self._quick_test(s))
            examples_layout.addWidget(btn)

        right_layout.addWidget(examples_group)

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
        """Restore saved column widths from settings."""
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
        """Save column widths to settings."""
        widths = [
            self.pattern_tree.columnWidth(0),
            self.pattern_tree.columnWidth(1),
            self.pattern_tree.columnWidth(2),
            self.pattern_tree.columnWidth(3),
        ]
        self.settings.set_custom_value('pattern_manager_columns_v3', widths)

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

    def _on_item_double_click(self, item, column):
        """Handle double-click on color or catalog column."""
        data = item.data(0, Qt.UserRole)
        if not data or data.get('is_library'):
            return

        name = data['name']

        if column == 3:
            # Color column - show color picker
            current_color = self.settings.get_pattern_color(name)
            initial = QColor(current_color) if current_color else QColor("#2e7d32")
            color = QColorDialog.getColor(initial, self, f"Choose color for {name}")

            if color.isValid():
                hex_color = color.name()
                self.settings.set_pattern_color(name, hex_color)
                item.setText(3, "●")
                item.setForeground(3, color)

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

            # Update button text based on editability
            if self._current_lua_editable:
                self.view_script_btn.setText("Edit Script")
                self.view_script_btn.setToolTip("Edit this Lua pattern script")
            else:
                self.view_script_btn.setText("View Script")
                self.view_script_btn.setToolTip("View the Lua source code (core patterns are read-only)")
        else:
            self._current_lua_pattern = None
            self._current_lua_editable = False
            self.lua_script_label.hide()
            self.view_script_btn.hide()

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

        # Create a dialog to show/edit the script
        dialog = QDialog(self)
        title = f"Edit Script: {self._current_lua_pattern}" if is_editable else f"View Script: {self._current_lua_pattern}"
        dialog.setWindowTitle(title)
        dialog.setMinimumSize(700, 500)

        layout = QVBoxLayout(dialog)

        # Info label
        if is_editable:
            info_text = (
                f"<b>{self._current_lua_pattern}</b> - {lua_info.description}<br>"
                f"<i>File: {lua_info.file_path}</i><br><br>"
                "Edit the script below and click Save to apply changes."
            )
        else:
            info_text = (
                f"<b>{self._current_lua_pattern}</b> - {lua_info.description}<br>"
                f"<i>File: {lua_info.file_path}</i><br><br>"
                "Core patterns are read-only. Use 'Create Copy' to make an editable version."
            )
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Script editor/viewer
        script_edit = QPlainTextEdit()
        script_edit.setReadOnly(not is_editable)
        script_edit.setPlainText(lua_info.script)
        font = QFont("Consolas, Monaco, monospace")
        font.setPointSize(11)
        script_edit.setFont(font)
        script_edit.setLineWrapMode(QPlainTextEdit.NoWrap)

        # Add syntax highlighting
        highlighter = LuaSyntaxHighlighter(script_edit.document())

        layout.addWidget(script_edit)

        # Status label for validation feedback
        status_label = QLabel("")
        status_label.setStyleSheet("color: gray;")
        layout.addWidget(status_label)

        # Buttons
        btn_layout = QHBoxLayout()

        if is_editable:
            # Validate button
            validate_btn = QPushButton("Validate")
            validate_btn.setToolTip("Check script syntax")
            validate_btn.clicked.connect(
                lambda: self._validate_script_in_dialog(script_edit, status_label)
            )
            btn_layout.addWidget(validate_btn)

            # Save button
            save_btn = QPushButton("Save")
            save_btn.setToolTip("Save changes to the script file")
            save_btn.clicked.connect(
                lambda: self._save_edited_script(lua_info, script_edit, status_label, dialog)
            )
            btn_layout.addWidget(save_btn)

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

    def _validate_script_in_dialog(self, script_edit, status_label):
        """Validate script syntax and update status label."""
        script = script_edit.toPlainText()
        valid, error = self.engine.validate_script(script)
        if valid:
            status_label.setText("✓ Syntax OK")
            status_label.setStyleSheet("color: green;")
        else:
            status_label.setText(f"✗ Error: {error}")
            status_label.setStyleSheet("color: red;")

    def _save_edited_script(self, lua_info, script_edit, status_label, dialog):
        """Save the edited script to the file."""
        script = script_edit.toPlainText()

        # Validate first
        valid, error = self.engine.validate_script(script)
        if not valid:
            status_label.setText(f"✗ Cannot save - syntax error: {error}")
            status_label.setStyleSheet("color: red;")
            return

        # Save to file
        try:
            with open(lua_info.file_path, 'w') as f:
                f.write(script)

            status_label.setText("✓ Saved successfully!")
            status_label.setStyleSheet("color: green;")

            # Reload the engine to pick up changes
            self.engine.reload()
            self._load_patterns()

            QMessageBox.information(
                dialog, "Saved",
                f"Script saved to:\n{lua_info.file_path}\n\nPatterns have been reloaded."
            )

        except Exception as e:
            status_label.setText(f"✗ Save failed: {e}")
            status_label.setStyleSheet("color: red;")
            QMessageBox.critical(dialog, "Save Error", f"Failed to save script:\n{e}")

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
                        display_name=defn.get('display_name', '')
                    )
                else:
                    self.engine.add_custom_pattern(name, defn)

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

    def _test_serial(self):
        """Test a serial number against all patterns."""
        serial = self.test_edit.text().strip()
        if not serial:
            return

        # Make sure it looks like a valid serial format
        if len(serial) < 10:
            # Pad with example format
            if len(serial) == 8 and serial.isdigit():
                serial = f"A{serial}B"

        matches = self.engine.classify(serial)

        if matches:
            result_text = f"Serial: {serial}\n\n"
            result_text += "Matches:\n"
            for match in matches:
                result_text += f"  - {match.name} (Tier {match.tier})\n"
                result_text += f"    {match.description}\n"
                # Get odds and price from pattern definition
                pattern_info = self.engine.get_pattern_info(match.name)
                if pattern_info:
                    if 'odds' in pattern_info:
                        result_text += f"    Odds: {pattern_info['odds']}\n"
                    if 'price_range' in pattern_info:
                        result_text += f"    Price: {pattern_info['price_range']}\n"
        else:
            result_text = f"Serial: {serial}\n\nNo patterns matched."

        self.test_results.setText(result_text)

    def _quick_test(self, serial: str):
        """Quick test with a predefined serial."""
        self.test_edit.setText(serial)
        self._test_serial()

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
        self.settings.save()
        self.accept()

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
                        library=library,
                        display_name=defn.get('display_name', '')
                    )
                else:
                    # Save as YAML rule pattern (legacy)
                    self.engine.add_custom_pattern(name, defn)

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
        self.digits = "12345678"
        self.highlights = []
        self.connectors = []
        self.setMinimumHeight(80)
        self.setMinimumWidth(400)

    def set_serial(self, serial: str):
        """Set the serial number to display."""
        self.digits = ''.join(c for c in serial if c.isdigit())
        if len(self.digits) != 8:
            self.digits = self.digits[:8].ljust(8, '0')
        self.update()

    def set_highlights(self, highlights: list, connectors: list):
        """Set highlights and connectors."""
        self.highlights = highlights or []
        self.connectors = connectors or []
        self.update()

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QPen, QBrush
        from PySide6.QtCore import QRect, QPoint

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Calculate box dimensions
        box_width = 40
        box_height = 50
        spacing = 8
        total_width = 8 * box_width + 7 * spacing
        start_x = (self.width() - total_width) // 2
        start_y = 20

        # Build color map for each position
        position_colors = {}
        for h in self.highlights:
            positions = h.get('positions', [])
            color = h.get('color', 'gray')
            for pos in positions:
                if 0 <= pos < 8:
                    position_colors[pos] = color

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

        # Draw digit boxes
        font = painter.font()
        font.setPointSize(16)
        font.setBold(True)
        painter.setFont(font)

        box_rects = []
        for i, digit in enumerate(self.digits):
            x = start_x + i * (box_width + spacing)
            rect = QRect(x, start_y, box_width, box_height)
            box_rects.append(rect)

            # Background color
            if i in position_colors:
                bg_color = color_map.get(position_colors[i], QColor("#9E9E9E"))
                painter.setBrush(QBrush(bg_color))
                painter.setPen(QPen(bg_color.darker(120), 2))
            else:
                painter.setBrush(QBrush(QColor("#2D2D2D")))
                painter.setPen(QPen(QColor("#555555"), 2))

            painter.drawRoundedRect(rect, 5, 5)

            # Draw digit
            painter.setPen(QPen(QColor("white")))
            painter.drawText(rect, Qt.AlignCenter, digit)

        # Draw connectors (arcs above the boxes)
        for conn in self.connectors:
            from_pos = conn.get('from', 0)
            to_pos = conn.get('to', 0)
            color = conn.get('color', 'gray')
            style = conn.get('style', 'arc')

            if 0 <= from_pos < 8 and 0 <= to_pos < 8:
                from_rect = box_rects[from_pos]
                to_rect = box_rects[to_pos]

                from_x = from_rect.center().x()
                to_x = to_rect.center().x()
                y = start_y - 5

                pen = QPen(color_map.get(color, QColor("#9E9E9E")), 2)
                if style == 'dashed':
                    pen.setStyle(Qt.DashLine)
                painter.setPen(pen)

                # Draw arc
                mid_x = (from_x + to_x) // 2
                arc_height = min(15, abs(to_pos - from_pos) * 4)

                from PySide6.QtGui import QPainterPath
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
        self.setMinimumSize(700, 600)

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

        self._setup_ui()
        if name:
            self._load_existing()

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

        # Tab 2: Lua Script (if v3 engine available)
        if HAS_V3_ENGINE:
            self.script_tab = self._create_script_tab()
            self.tab_widget.addTab(self.script_tab, "Lua Script")

            # Tab 3: Test/Preview
            self.test_tab = self._create_test_tab()
            self.tab_widget.addTab(self.test_tab, "Test")

            # Tab 4: Documentation
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

        # Test input
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

        layout.addLayout(input_layout)

        # Preview widget
        preview_group = QGroupBox("Visual Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.digit_preview = DigitPreviewWidget()
        preview_layout.addWidget(self.digit_preview)

        layout.addWidget(preview_group)

        # Results
        results_group = QGroupBox("Test Results")
        results_layout = QVBoxLayout(results_group)

        self.test_results = QTextEdit()
        self.test_results.setReadOnly(True)
        self.test_results.setMaximumHeight(150)
        results_layout.addWidget(self.test_results)

        layout.addWidget(results_group)

        layout.addStretch()

        return widget

    def _create_docs_tab(self) -> QWidget:
        """Create the API documentation tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Copy button
        copy_btn = QPushButton("Copy API Docs to Clipboard")
        copy_btn.clicked.connect(self._copy_api_docs)
        layout.addWidget(copy_btn)

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

    def _update_hint(self):
        """Update the hint based on rule type."""
        rule = self.rule_type.currentText()
        hints = {
            "contains": "Matches if serial contains this value anywhere.\nExamples: '0704' matches July 4, '1990' matches birth year",
            "starts_with": "Matches if serial starts with this value.\nExample: '000' matches low serial numbers",
            "ends_with": "Matches if serial ends with this value.\nExample: '0000' matches round numbers",
            "regex": "Advanced: Regular expression pattern.\nExample: '(\\d)\\1{3}' matches 4 repeated digits",
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

        # Check which tab is active
        if self.tab_widget.currentIndex() == 1:  # Script tab
            script = self.script_edit.toPlainText()
            result = self.engine.test_script(script, serial)

            if result.success:
                self.digit_preview.set_highlights(result.highlights, result.connectors)

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
        else:
            # Simple rule test
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

        # Check which mode we're in
        if HAS_V3_ENGINE and self.tab_widget.currentIndex() == 1:
            # Script mode - validate syntax
            script = self.script_edit.toPlainText()
            valid, error = self.engine.validate_script(script)
            if not valid:
                QMessageBox.warning(self, "Script Error", f"Lua syntax error:\n{error}")
                return
            self.is_lua_pattern = True
        else:
            # Simple rule mode
            value = self.value_edit.text().strip()
            if not value:
                QMessageBox.warning(self, "Validation Error", "Please enter a value to match.")
                return
            self.is_lua_pattern = False

        self.accept()

    def get_pattern(self) -> tuple:
        """Return the pattern name and definition."""
        name = self.name_edit.text().strip().upper().replace(' ', '_')
        tier = self.tier_spin.value()
        description = self.desc_edit.text().strip() or f"Custom pattern: {name}"
        display_name = self.display_name_edit.text().strip()

        if self.is_lua_pattern and HAS_V3_ENGINE:
            # Return script
            script = self.script_edit.toPlainText()
            return name, {
                'description': description,
                'display_name': display_name,
                'tier': tier,
                'script': script,
                'source': 'lua'
            }
        else:
            # Return simple rule
            rule_type = self.rule_type.currentText()
            value = self.value_edit.text().strip()

            return name, {
                'description': description,
                'display_name': display_name,
                'tier': tier,
                'enabled': True,
                'rules': {rule_type: value},
                'source': 'yaml'
            }


# Test dialog standalone
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dialog = PatternDialog()
    dialog.exec()
