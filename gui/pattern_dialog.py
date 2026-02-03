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

# Try to import v3 engine, fall back to v2
try:
    from pattern_engine_v3 import PatternEngineV3 as PatternEngine
    HAS_V3_ENGINE = True
except ImportError:
    from pattern_engine_v2 import PatternEngine
    HAS_V3_ENGINE = False

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

        # Pattern tree
        self.pattern_tree = QTreeWidget()
        self.pattern_tree.setHeaderLabels(["Pattern", "Tier", "Enabled", "Color", "Catalog"])
        self.pattern_tree.setRootIsDecorated(True)
        self.pattern_tree.itemChanged.connect(self._on_item_changed)
        self.pattern_tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.pattern_tree.itemDoubleClicked.connect(self._on_item_double_click)

        header = self.pattern_tree.header()
        header.setSectionsMovable(False)  # Don't allow reordering columns
        header.setStretchLastSection(True)  # Last column fills remaining space
        # All columns interactive (draggable) except last which stretches
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.Interactive)
        header.setSectionResizeMode(3, QHeaderView.Interactive)
        header.setSectionResizeMode(4, QHeaderView.Stretch)

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

        # Custom patterns section
        custom_group = QGroupBox("Custom Patterns (Birthdays, Anniversaries, etc.)")
        custom_layout = QVBoxLayout(custom_group)

        self.custom_list = QListWidget()
        self.custom_list.itemSelectionChanged.connect(self._on_custom_selection_changed)
        custom_layout.addWidget(self.custom_list)

        custom_btn_layout = QHBoxLayout()
        add_custom_btn = QPushButton("Add...")
        add_custom_btn.clicked.connect(self._add_custom_pattern)
        custom_btn_layout.addWidget(add_custom_btn)

        edit_custom_btn = QPushButton("Edit...")
        edit_custom_btn.clicked.connect(self._edit_custom_pattern)
        custom_btn_layout.addWidget(edit_custom_btn)

        delete_custom_btn = QPushButton("Delete")
        delete_custom_btn.clicked.connect(self._delete_custom_pattern)
        custom_btn_layout.addWidget(delete_custom_btn)

        custom_layout.addLayout(custom_btn_layout)

        right_layout.addWidget(custom_group)

        # Load custom patterns
        self._load_custom_patterns()

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
        widths = self.settings.get_custom_value('pattern_manager_columns', None)
        if widths and len(widths) >= 4:
            self.pattern_tree.setColumnWidth(0, widths[0])  # Pattern
            self.pattern_tree.setColumnWidth(1, widths[1])  # Tier
            self.pattern_tree.setColumnWidth(2, widths[2])  # Enabled
            self.pattern_tree.setColumnWidth(3, widths[3])  # Color
            # Column 4 (Catalog) stretches to fill
        else:
            # Default widths
            self.pattern_tree.setColumnWidth(0, 200)  # Pattern
            self.pattern_tree.setColumnWidth(1, 50)   # Tier
            self.pattern_tree.setColumnWidth(2, 60)   # Enabled
            self.pattern_tree.setColumnWidth(3, 50)   # Color

    def _save_column_widths(self):
        """Save column widths to settings."""
        widths = [
            self.pattern_tree.columnWidth(0),
            self.pattern_tree.columnWidth(1),
            self.pattern_tree.columnWidth(2),
            self.pattern_tree.columnWidth(3),
        ]
        self.settings.set_custom_value('pattern_manager_columns', widths)

    def _load_patterns(self):
        """Load patterns into the tree."""
        self.pattern_tree.clear()

        # Get Lua pattern names for indicator
        lua_pattern_names = set()
        if HAS_V3_ENGINE and hasattr(self.engine, 'lua_patterns'):
            lua_pattern_names = set(self.engine.lua_patterns.keys())

        # Group by tier
        tiers = {}
        all_patterns = self.engine.config.get('patterns', {})

        for name, defn in all_patterns.items():
            if defn is None:
                continue
            tier = defn.get('tier', 10)
            if tier not in tiers:
                tiers[tier] = []
            tiers[tier].append((name, defn))

        # Add to tree
        tier_names = {
            1: "Tier 1: Holy Grail ($500+)",
            2: "Tier 2: Premium ($100-500)",
            3: "Tier 3: Collector ($20-100)",
            4: "Tier 4: Interesting ($5-20)",
            5: "Tier 5: Sum Patterns",
            6: "Tier 6: Ladder Variants",
            7: "Tier 7: Flipper Patterns",
            8: "Tier 8: Low Serial Variants",
            9: "Tier 9: Structural Combos",
            10: "Tier 10: Novelty",
        }

        for tier in sorted(tiers.keys()):
            tier_item = QTreeWidgetItem()
            tier_item.setText(0, tier_names.get(tier, f"Tier {tier}"))
            tier_item.setData(0, Qt.UserRole, {'is_tier': True, 'tier': tier})
            tier_item.setFlags(tier_item.flags() & ~Qt.ItemIsSelectable)

            for name, defn in sorted(tiers[tier], key=lambda x: x[0]):
                pattern_item = QTreeWidgetItem(tier_item)

                # Add [Lua] indicator if pattern has Lua implementation
                has_lua = name in lua_pattern_names
                display_name = f"{name} [Lua]" if has_lua else name
                pattern_item.setText(0, display_name)
                pattern_item.setText(1, str(tier))

                # Checkbox for enabled - check if pattern is active (considers user overrides)
                enabled = name in self.engine.patterns
                pattern_item.setCheckState(2, Qt.Checked if enabled else Qt.Unchecked)
                pattern_item.setData(0, Qt.UserRole, {'name': name, 'defn': defn, 'has_lua': has_lua})

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

            self.pattern_tree.addTopLevelItem(tier_item)

        self.pattern_tree.expandAll()

    def _filter_patterns(self):
        """Filter patterns based on search text."""
        filter_text = self.filter_edit.text().lower()
        show_disabled = self.show_disabled_check.isChecked()

        for i in range(self.pattern_tree.topLevelItemCount()):
            tier_item = self.pattern_tree.topLevelItem(i)
            visible_children = 0

            for j in range(tier_item.childCount()):
                pattern_item = tier_item.child(j)
                data = pattern_item.data(0, Qt.UserRole)
                name = data['name'].lower()
                desc = data['defn'].get('description', '').lower()
                enabled = pattern_item.checkState(2) == Qt.Checked

                # Filter by text
                text_match = not filter_text or filter_text in name or filter_text in desc

                # Filter by enabled state
                enabled_match = show_disabled or enabled

                visible = text_match and enabled_match
                pattern_item.setHidden(not visible)

                if visible:
                    visible_children += 1

            tier_item.setHidden(visible_children == 0)

    def _on_item_changed(self, item, column):
        """Handle item check state change."""
        if column != 2:
            return

        data = item.data(0, Qt.UserRole)
        if not data or data.get('is_tier'):
            return

        name = data['name']
        enabled = item.checkState(2) == Qt.Checked
        self.engine.set_pattern_enabled(name, enabled)

    def _on_item_double_click(self, item, column):
        """Handle double-click on color or catalog column."""
        data = item.data(0, Qt.UserRole)
        if not data or data.get('is_tier'):
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
        if not data or data.get('is_tier'):
            return

        name = data['name']
        defn = data['defn']
        has_lua = data.get('has_lua', False)

        # Check if there's a Lua implementation with more details
        lua_info = None
        if has_lua and HAS_V3_ENGINE and hasattr(self.engine, 'lua_patterns'):
            lua_info = self.engine.lua_patterns.get(name)

        self.pattern_name_label.setText(name)

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

        # Show/hide Lua script viewer
        if has_lua and lua_info:
            self._current_lua_pattern = name
            self.lua_script_label.show()
            self.view_script_btn.show()
        else:
            self._current_lua_pattern = None
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
        # Get the rule type from the pattern definition
        all_patterns = self.engine.config.get('patterns', {})
        defn = all_patterns.get(name, {})
        rules = defn.get('rules', {})

        rule_type = None
        for rt in ['baseline_variance_min', 'baseline_variance_max']:
            if rt in rules:
                rule_type = rt
                break

        if not rule_type:
            return

        # Store override in SettingsManager
        self.settings.set_pattern_override(name, rule_type, value)
        self.settings.save()

        # Rebuild patterns to pick up the new threshold
        self.engine.reload()

        QMessageBox.information(self, "Saved", f"Threshold for {name} set to {value}")

    def _view_lua_script(self):
        """View the Lua script for the selected pattern."""
        if not self._current_lua_pattern:
            return

        if not HAS_V3_ENGINE or not hasattr(self.engine, 'lua_patterns'):
            return

        lua_info = self.engine.lua_patterns.get(self._current_lua_pattern)
        if not lua_info:
            return

        # Create a dialog to show the script
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Lua Script: {self._current_lua_pattern}")
        dialog.setMinimumSize(700, 500)

        layout = QVBoxLayout(dialog)

        # Info label
        info_label = QLabel(
            f"<b>{self._current_lua_pattern}</b> - {lua_info.description}<br>"
            f"<i>File: {lua_info.file_path}</i><br><br>"
            "You can copy this script and use it as a template for your own pattern."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Script viewer
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
        copy_btn.clicked.connect(lambda: self._copy_script_to_clipboard(lua_info.script))
        btn_layout.addWidget(copy_btn)

        create_copy_btn = QPushButton("Create Modified Copy...")
        create_copy_btn.setToolTip("Create a new user pattern based on this script")
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

        # Open CustomPatternDialog pre-filled with the script
        dialog = CustomPatternDialog(
            self,
            name=f"{lua_info.name}_CUSTOM",
            defn={
                'description': f"Modified version of {lua_info.name}",
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
                        defn.get('tier', 5)
                    )
                else:
                    self.engine.add_custom_pattern(name, defn)
                self._load_custom_patterns()
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
        """Enable all patterns."""
        for i in range(self.pattern_tree.topLevelItemCount()):
            tier_item = self.pattern_tree.topLevelItem(i)
            for j in range(tier_item.childCount()):
                pattern_item = tier_item.child(j)
                pattern_item.setCheckState(2, Qt.Checked)

    def _disable_all(self):
        """Disable all patterns."""
        for i in range(self.pattern_tree.topLevelItemCount()):
            tier_item = self.pattern_tree.topLevelItem(i)
            for j in range(tier_item.childCount()):
                pattern_item = tier_item.child(j)
                pattern_item.setCheckState(2, Qt.Unchecked)

    def _save_and_close(self):
        """Save pattern states and close."""
        self.engine.save_config()
        self.settings.save()  # Save colors
        self.accept()

    def _load_custom_patterns(self):
        """Load custom patterns into the list."""
        self.custom_list.clear()

        # Load YAML custom patterns
        custom = self.engine.get_custom_patterns()
        if custom:
            for name, defn in custom.items():
                if defn is None:
                    continue
                desc = defn.get('description', '')
                rules = defn.get('rules', {})
                value = rules.get('contains', rules.get('regex', rules.get('starts_with', rules.get('ends_with', ''))))
                enabled = defn.get('enabled', True)

                item = QListWidgetItem()
                item.setText(f"[YAML] {name}: {desc}")
                item.setData(Qt.UserRole, {'name': name, 'defn': defn, 'script': None})
                item.setCheckState(Qt.Checked if enabled else Qt.Unchecked)
                self.custom_list.addItem(item)

        # Load Lua user patterns (if v3 engine available)
        if HAS_V3_ENGINE and hasattr(self.engine, 'get_user_patterns'):
            user_lua = self.engine.get_user_patterns()
            for name, info in user_lua.items():
                item = QListWidgetItem()
                item.setText(f"[Lua] {name}: {info.description}")
                item.setData(Qt.UserRole, {
                    'name': name,
                    'defn': {
                        'description': info.description,
                        'tier': info.tier,
                    },
                    'script': info.script
                })
                item.setCheckState(Qt.Checked if info.enabled else Qt.Unchecked)
                self.custom_list.addItem(item)

    def _on_custom_selection_changed(self):
        """Handle custom pattern selection change."""
        pass  # Could show details if needed

    def _add_custom_pattern(self):
        """Add a new custom pattern."""
        dialog = CustomPatternDialog(self)
        if dialog.exec() == QDialog.Accepted:
            name, defn = dialog.get_pattern()
            if name:
                if defn.get('source') == 'lua' and HAS_V3_ENGINE:
                    # Save as Lua script pattern
                    self.engine.save_user_pattern(
                        name,
                        defn.get('script', ''),
                        defn.get('description', ''),
                        defn.get('tier', 5)
                    )
                else:
                    # Save as YAML rule pattern
                    self.engine.add_custom_pattern(name, defn)
                self._load_custom_patterns()

    def _edit_custom_pattern(self):
        """Edit the selected custom pattern."""
        item = self.custom_list.currentItem()
        if not item:
            QMessageBox.information(self, "Edit Pattern", "Please select a pattern to edit.")
            return

        data = item.data(Qt.UserRole)
        name = data['name']
        defn = data['defn']
        script = data.get('script')

        dialog = CustomPatternDialog(self, name, defn, script)
        if dialog.exec() == QDialog.Accepted:
            new_name, new_defn = dialog.get_pattern()
            if new_name:
                # Remove old pattern if name changed
                if new_name != name:
                    if script and HAS_V3_ENGINE:
                        self.engine.delete_user_pattern(name)
                    else:
                        self.engine.remove_custom_pattern(name)

                # Add updated pattern
                if new_defn.get('source') == 'lua' and HAS_V3_ENGINE:
                    self.engine.save_user_pattern(
                        new_name,
                        new_defn.get('script', ''),
                        new_defn.get('description', ''),
                        new_defn.get('tier', 5)
                    )
                else:
                    self.engine.add_custom_pattern(new_name, new_defn)
                self._load_custom_patterns()

    def _delete_custom_pattern(self):
        """Delete the selected custom pattern."""
        item = self.custom_list.currentItem()
        if not item:
            QMessageBox.information(self, "Delete Pattern", "Please select a pattern to delete.")
            return

        data = item.data(Qt.UserRole)
        name = data['name']
        script = data.get('script')

        reply = QMessageBox.question(
            self, "Delete Pattern",
            f"Are you sure you want to delete '{name}'?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if script and HAS_V3_ENGINE:
                self.engine.delete_user_pattern(name)
            else:
                self.engine.remove_custom_pattern(name)
            self._load_custom_patterns()


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
        header_layout.addRow("Pattern Name:", self.name_edit)

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

<h3>Input Context (ctx)</h3>
<pre>
ctx.digits      -- "12345678" (8 numeric characters)
ctx.full_serial -- "A12345678B" (with prefix/suffix letters)
ctx.digit_list  -- {1,2,3,4,5,6,7,8} as integer array
ctx.metadata    -- {} additional detection data
</pre>

<h3>Return Value</h3>
<pre>
return {
    matched = true,  -- or false
    highlights = {
        {positions = {0, 7}, color = "orange", label = "pair"},
        {positions = {1, 6}, color = "coral"},
    },
    connectors = {
        {from = 0, to = 7, color = "orange", style = "arc"},
    },
    message = "Optional description"
}
</pre>

<h3>Available Colors</h3>
<ul>
<li><b>purple</b> - Flipper-valid digits (0,1,6,8,9)</li>
<li><b>blue</b> - Binary patterns (0,1)</li>
<li><b>cyan</b> - Trinary patterns</li>
<li><b>orange</b> - Primary pairs (radar)</li>
<li><b>coral</b> - Secondary pairs</li>
<li><b>gold</b> - Quads/runs</li>
<li><b>salmon</b> - Tertiary pairs</li>
<li><b>magenta</b> - Repeater</li>
<li><b>yellow</b> - Solid/dominant</li>
<li><b>lime</b> - Ladder sequence</li>
<li><b>teal</b> - Double pairs</li>
<li><b>red</b> - Errors/broken patterns</li>
</ul>

<h3>Connector Styles</h3>
<ul>
<li><b>arc</b> - Curved line above digits</li>
<li><b>line</b> - Straight line</li>
<li><b>dashed</b> - Dashed line</li>
<li><b>bracket</b> - Bracket connector</li>
</ul>

<h3>Helper Functions</h3>
<pre>
count_digits(s)           -- {["0"]=2, ["1"]=3, ...}
find_runs(s)              -- {{digit, start, length}, ...}
only_digits(s, allowed)   -- true if s contains only allowed
is_ladder(s)              -- true if ascending or descending
is_palindrome(s)          -- true if palindrome
most_common(s)            -- digit, count
unique_count(s)           -- number of unique digits
digit_sum(s)              -- sum of all digits
highlight(positions, color, label)
connector(from, to, color, style)
</pre>

<h3>Example: Palindrome Pattern</h3>
<pre>
function match(ctx)
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
        table.insert(connectors, {from = i, to = j, color = colors[i+1]})
    end

    return {matched = true, highlights = highlights, connectors = connectors}
end
</pre>
'''

    def _copy_api_docs(self):
        """Copy API docs to clipboard for pasting into AI chat."""
        docs = '''# Lua Pattern Script API for Dollar Bill Serial Numbers

## Input Context
The `ctx` table is available in every pattern script:
- ctx.digits: "12345678" (8 numeric characters)
- ctx.full_serial: "A12345678B" (with prefix/suffix letters)
- ctx.digit_list: {1,2,3,4,5,6,7,8} as integer array (1-indexed)
- ctx.metadata: {} additional detection metadata

## Return Value
The match function must return a table with:
- matched: boolean (true if pattern matches)
- highlights: list of {positions = {0, 7}, color = "orange", label = "optional"}
- connectors: list of {from = 0, to = 7, color = "orange", style = "arc"}
- message: optional string describing the match

## Available Colors
purple, blue, cyan, orange, coral, gold, salmon, magenta, yellow, lime, teal, red

## Helper Functions
- count_digits(s): returns table of digit counts
- find_runs(s): finds consecutive runs of same digit
- only_digits(s, allowed): checks if s contains only allowed digits
- is_ladder(s), is_ascending(s), is_descending(s): ladder checks
- is_palindrome(s): palindrome check
- most_common(s): returns most common digit and count
- unique_count(s): number of unique digits
- digit_sum(s): sum of all digits
- all_flip_valid(s): checks if all digits are flip-valid (0,1,6,8,9)
- flip_string(s): returns flipped version

## Example Pattern
```lua
function match(ctx)
    -- Check if palindrome
    local rev = string.reverse(ctx.digits)
    if ctx.digits ~= rev then
        return {matched = false}
    end

    local highlights = {}
    local connectors = {}
    local colors = {"orange", "coral", "gold", "salmon"}

    for i = 0, 3 do
        local j = 7 - i
        table.insert(highlights, {positions = {i, j}, color = colors[i+1]})
        table.insert(connectors, {from = i, to = j, color = colors[i+1], style = "arc"})
    end

    return {matched = true, highlights = highlights, connectors = connectors}
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

    def _load_existing(self):
        """Load existing pattern data."""
        self.name_edit.setText(self.original_name)
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

        if self.is_lua_pattern and HAS_V3_ENGINE:
            # Return script
            script = self.script_edit.toPlainText()
            return name, {
                'description': description,
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
