"""
Pattern Dialog - Manage pattern enable/disable and testing.
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QGroupBox, QLineEdit, QPushButton, QDialogButtonBox, QLabel,
    QTextEdit, QSplitter, QHeaderView, QCheckBox, QListWidget,
    QListWidgetItem, QFormLayout, QComboBox, QMessageBox
)
from PySide6.QtCore import Qt

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from pattern_engine_v2 import PatternEngine


class PatternDialog(QDialog):
    """Dialog for managing patterns and testing serials."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = PatternEngine()

        self.setWindowTitle("Pattern Manager")
        self.setMinimumSize(800, 600)
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
        self.pattern_tree.setHeaderLabels(["Pattern", "Tier", "Enabled"])
        self.pattern_tree.setRootIsDecorated(True)
        self.pattern_tree.itemChanged.connect(self._on_item_changed)
        self.pattern_tree.itemSelectionChanged.connect(self._on_selection_changed)

        header = self.pattern_tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)

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

    def _load_patterns(self):
        """Load patterns into the tree."""
        self.pattern_tree.clear()

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
                pattern_item.setText(0, name)
                pattern_item.setText(1, str(tier))

                # Checkbox for enabled
                enabled = defn.get('enabled', True)
                pattern_item.setCheckState(2, Qt.Checked if enabled else Qt.Unchecked)
                pattern_item.setData(0, Qt.UserRole, {'name': name, 'defn': defn})

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

        self.pattern_name_label.setText(name)
        self.pattern_desc_label.setText(defn.get('description', 'No description'))
        self.pattern_tier_label.setText(f"Tier: {defn.get('tier', '?')}")

        examples = defn.get('examples', [])
        if examples:
            self.pattern_examples_label.setText(f"Examples: {', '.join(examples)}")
        else:
            self.pattern_examples_label.setText("Examples: (none)")

        odds = defn.get('odds', '')
        if odds:
            self.pattern_odds_label.setText(f"Odds: {odds}")
        else:
            self.pattern_odds_label.setText("Odds: (not calculated)")

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
                # Get odds from pattern definition
                pattern_info = self.engine.get_pattern_info(match.name)
                if pattern_info and 'odds' in pattern_info:
                    result_text += f"    Odds: {pattern_info['odds']}\n"
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
        self.accept()

    def _load_custom_patterns(self):
        """Load custom patterns into the list."""
        self.custom_list.clear()
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
                item.setText(f"{name}: {desc}")
                item.setData(Qt.UserRole, {'name': name, 'defn': defn})
                item.setCheckState(Qt.Checked if enabled else Qt.Unchecked)
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

        dialog = CustomPatternDialog(self, name, defn)
        if dialog.exec() == QDialog.Accepted:
            new_name, new_defn = dialog.get_pattern()
            if new_name:
                # Remove old pattern if name changed
                if new_name != name:
                    self.engine.remove_custom_pattern(name)
                # Add updated pattern
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

        reply = QMessageBox.question(
            self, "Delete Pattern",
            f"Are you sure you want to delete '{name}'?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.engine.remove_custom_pattern(name)
            self._load_custom_patterns()


class CustomPatternDialog(QDialog):
    """Dialog for adding/editing a custom pattern."""

    def __init__(self, parent=None, name: str = "", defn: dict = None):
        super().__init__(parent)
        self.setWindowTitle("Add Custom Pattern" if not name else "Edit Custom Pattern")
        self.setMinimumWidth(400)

        self.original_name = name
        self.defn = defn or {}

        self._setup_ui()
        if name:
            self._load_existing()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)

        form = QFormLayout()

        # Pattern name
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., MY_BIRTHDAY, ANNIVERSARY, LUCKY_NUMBER")
        form.addRow("Pattern Name:", self.name_edit)

        # Description
        self.desc_edit = QLineEdit()
        self.desc_edit.setPlaceholderText("e.g., Mom's birthday (July 4)")
        form.addRow("Description:", self.desc_edit)

        # Rule type
        self.rule_type = QComboBox()
        self.rule_type.addItems(["contains", "starts_with", "ends_with", "regex"])
        self.rule_type.setCurrentText("contains")
        self.rule_type.currentTextChanged.connect(self._update_hint)
        form.addRow("Rule Type:", self.rule_type)

        # Value
        self.value_edit = QLineEdit()
        self.value_edit.setPlaceholderText("e.g., 0704 for July 4th")
        form.addRow("Value:", self.value_edit)

        # Hint label
        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("color: gray; font-style: italic;")
        self._update_hint()
        form.addRow("", self.hint_label)

        # Tier (default to 10 for custom)
        self.tier_combo = QComboBox()
        self.tier_combo.addItems(["10 - Novelty", "9 - Structural", "4 - Interesting", "3 - Collector"])
        form.addRow("Tier:", self.tier_combo)

        layout.addLayout(form)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _update_hint(self):
        """Update the hint based on rule type."""
        rule = self.rule_type.currentText()
        hints = {
            "contains": "Matches if serial contains this value anywhere.\nExamples: '0704' matches July 4, '1990' matches birth year",
            "starts_with": "Matches if serial starts with this value.\nExample: '000' matches low serial numbers",
            "ends_with": "Matches if serial ends with this value.\nExample: '0000' matches round numbers",
            "regex": "Advanced: Regular expression pattern.\nExample: '(\\d)\\1{3}' matches 4 repeated digits"
        }
        self.hint_label.setText(hints.get(rule, ""))

    def _load_existing(self):
        """Load existing pattern data."""
        self.name_edit.setText(self.original_name)
        self.desc_edit.setText(self.defn.get('description', ''))

        rules = self.defn.get('rules', {})
        for rule_type in ['contains', 'starts_with', 'ends_with', 'regex']:
            if rule_type in rules:
                self.rule_type.setCurrentText(rule_type)
                self.value_edit.setText(str(rules[rule_type]))
                break

        tier = self.defn.get('tier', 10)
        tier_map = {10: 0, 9: 1, 4: 2, 3: 3}
        self.tier_combo.setCurrentIndex(tier_map.get(tier, 0))

    def _validate_and_accept(self):
        """Validate input and accept."""
        name = self.name_edit.text().strip().upper().replace(' ', '_')
        if not name:
            QMessageBox.warning(self, "Validation Error", "Please enter a pattern name.")
            return

        value = self.value_edit.text().strip()
        if not value:
            QMessageBox.warning(self, "Validation Error", "Please enter a value to match.")
            return

        self.accept()

    def get_pattern(self) -> tuple:
        """Return the pattern name and definition."""
        name = self.name_edit.text().strip().upper().replace(' ', '_')
        rule_type = self.rule_type.currentText()
        value = self.value_edit.text().strip()

        tier_text = self.tier_combo.currentText()
        tier = int(tier_text.split(' ')[0])

        defn = {
            'description': self.desc_edit.text().strip() or f"Custom pattern: {value}",
            'tier': tier,
            'enabled': True,
            'rules': {rule_type: value}
        }

        return name, defn


# Test dialog standalone
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dialog = PatternDialog()
    dialog.exec()
