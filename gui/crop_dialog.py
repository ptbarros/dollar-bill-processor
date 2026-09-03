"""
eBay Crop Manager Dialog - Configure crop regions and output order.
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QGroupBox, QPushButton, QDialogButtonBox, QLabel, QSpinBox,
    QHeaderView, QCheckBox, QAbstractItemView, QMessageBox, QWidget, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class EbayCropDialog(QDialog):
    """Dialog for managing eBay crop settings and order."""

    # Default crops with their properties
    DEFAULT_CROPS = [
        {'side': 'front', 'region': 'seal', 'name': 'Front Seal', 'has_settings': True},
        {'side': 'front', 'region': 'full', 'name': 'Front Full', 'has_settings': False},
        {'side': 'front', 'region': 'left', 'name': 'Front Left', 'has_settings': False},
        {'side': 'front', 'region': 'center', 'name': 'Front Center', 'has_settings': False},
        {'side': 'front', 'region': 'right', 'name': 'Front Right', 'has_settings': False},
        {'side': 'back', 'region': 'seal', 'name': 'Back Seal', 'has_settings': True},
        {'side': 'back', 'region': 'full', 'name': 'Back Full', 'has_settings': False},
        {'side': 'back', 'region': 'left', 'name': 'Back Left', 'has_settings': False},
        {'side': 'back', 'region': 'center', 'name': 'Back Center', 'has_settings': False},
        {'side': 'back', 'region': 'right', 'name': 'Back Right', 'has_settings': False},
    ]

    def __init__(self, config, parent=None, preview_ctx=None):
        super().__init__(parent)
        self.config = config
        self.preview_ctx = preview_ctx
        self.setWindowTitle("eBay Crop Manager")
        self.setMinimumSize(1080 if preview_ctx else 700, 560)
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Setup the dialog UI."""
        outer = QHBoxLayout(self)
        layout = QVBoxLayout()
        outer.addLayout(layout, 1)

        # Instructions
        instructions = QLabel(
            "Configure which crops are generated and their output order.\n"
            "The order number determines the filename suffix (e.g., _01.jpg, _02.jpg).\n"
            + ("Select a crop to preview it on a sample bill; seal crops have size "
               "and offset settings." if self.preview_ctx else
               "Seal crops have additional size and offset settings.")
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Crop order table
        order_group = QGroupBox("Crop Order")
        order_layout = QVBoxLayout(order_group)

        self.crop_table = QTableWidget()
        self.crop_table.setColumnCount(4)
        self.crop_table.setHorizontalHeaderLabels(["Enabled", "Crop", "Order", "Settings"])
        self.crop_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.crop_table.setSelectionMode(QAbstractItemView.SingleSelection)

        header = self.crop_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)

        order_layout.addWidget(self.crop_table)

        # Move up/down buttons
        btn_layout = QHBoxLayout()
        move_up_btn = QPushButton("Move Up")
        move_up_btn.clicked.connect(self._move_up)
        btn_layout.addWidget(move_up_btn)

        move_down_btn = QPushButton("Move Down")
        move_down_btn.clicked.connect(self._move_down)
        btn_layout.addWidget(move_down_btn)

        btn_layout.addStretch()

        reset_order_btn = QPushButton("Reset to Default Order")
        reset_order_btn.clicked.connect(self._reset_order)
        btn_layout.addWidget(reset_order_btn)

        order_layout.addLayout(btn_layout)
        layout.addWidget(order_group)

        # Front Seal Settings
        front_seal_group = QGroupBox("Front Seal Settings")
        self.front_seal_group = front_seal_group
        front_seal_layout = QHBoxLayout(front_seal_group)

        front_seal_layout.addWidget(QLabel("Width:"))
        self.front_seal_width = QSpinBox()
        self.front_seal_width.setRange(100, 2000)
        self.front_seal_width.setSuffix(" px")
        front_seal_layout.addWidget(self.front_seal_width)

        front_seal_layout.addWidget(QLabel("Height:"))
        self.front_seal_height = QSpinBox()
        self.front_seal_height.setRange(100, 2000)
        self.front_seal_height.setSuffix(" px")
        front_seal_layout.addWidget(self.front_seal_height)

        front_seal_layout.addWidget(QLabel("Offset X:"))
        self.front_seal_offset_x = QSpinBox()
        self.front_seal_offset_x.setRange(-500, 500)
        self.front_seal_offset_x.setSuffix(" px")
        self.front_seal_offset_x.setToolTip("Positive = shift right, Negative = shift left")
        front_seal_layout.addWidget(self.front_seal_offset_x)

        front_seal_layout.addWidget(QLabel("Offset Y:"))
        self.front_seal_offset_y = QSpinBox()
        self.front_seal_offset_y.setRange(-500, 500)
        self.front_seal_offset_y.setSuffix(" px")
        self.front_seal_offset_y.setToolTip("Positive = shift up, Negative = shift down")
        front_seal_layout.addWidget(self.front_seal_offset_y)

        front_seal_layout.addStretch()
        layout.addWidget(front_seal_group)

        # Back Seal Settings
        back_seal_group = QGroupBox("Back Seal Settings")
        self.back_seal_group = back_seal_group
        back_seal_layout = QHBoxLayout(back_seal_group)

        back_seal_layout.addWidget(QLabel("Width:"))
        self.back_seal_width = QSpinBox()
        self.back_seal_width.setRange(100, 2000)
        self.back_seal_width.setSuffix(" px")
        back_seal_layout.addWidget(self.back_seal_width)

        back_seal_layout.addWidget(QLabel("Height:"))
        self.back_seal_height = QSpinBox()
        self.back_seal_height.setRange(100, 2000)
        self.back_seal_height.setSuffix(" px")
        back_seal_layout.addWidget(self.back_seal_height)

        back_seal_layout.addWidget(QLabel("Offset X:"))
        self.back_seal_offset_x = QSpinBox()
        self.back_seal_offset_x.setRange(-500, 500)
        self.back_seal_offset_x.setSuffix(" px")
        self.back_seal_offset_x.setToolTip("Positive = shift right, Negative = shift left")
        back_seal_layout.addWidget(self.back_seal_offset_x)

        back_seal_layout.addWidget(QLabel("Offset Y:"))
        self.back_seal_offset_y = QSpinBox()
        self.back_seal_offset_y.setRange(-500, 500)
        self.back_seal_offset_y.setSuffix(" px")
        self.back_seal_offset_y.setToolTip("Positive = shift up, Negative = shift down")
        back_seal_layout.addWidget(self.back_seal_offset_y)

        back_seal_layout.addStretch()
        layout.addWidget(back_seal_group)

        # Third-crop overlap adjustments (contextual: shown for the selected
        # left/center/right crop). Lets a boundary-straddling defect be pulled
        # fully into one third by overlapping into the center.
        self.thirds_group = QGroupBox("Third-Crop Overlap")
        tg = QHBoxLayout(self.thirds_group)
        self.lbl_left_inner = QLabel("Extend inner edge right:")
        tg.addWidget(self.lbl_left_inner)
        self.left_inner = QSpinBox()
        self.left_inner.setRange(-1000, 1000)
        self.left_inner.setSuffix(" px")
        self.left_inner.setToolTip("Grow the Left crop's inner (right) edge toward "
                                   "center. Positive = more overlap with the center crop.")
        tg.addWidget(self.left_inner)
        self.lbl_right_inner = QLabel("Extend inner edge left:")
        tg.addWidget(self.lbl_right_inner)
        self.right_inner = QSpinBox()
        self.right_inner.setRange(-1000, 1000)
        self.right_inner.setSuffix(" px")
        self.right_inner.setToolTip("Grow the Right crop's inner (left) edge toward "
                                    "center. Positive = more overlap with the center crop.")
        tg.addWidget(self.right_inner)
        self.lbl_center_left = QLabel("Expand left:")
        tg.addWidget(self.lbl_center_left)
        self.center_left = QSpinBox()
        self.center_left.setRange(-1000, 1000)
        self.center_left.setSuffix(" px")
        self.center_left.setToolTip("Expand the Center crop's left edge outward.")
        tg.addWidget(self.center_left)
        self.lbl_center_right = QLabel("Expand right:")
        tg.addWidget(self.lbl_center_right)
        self.center_right = QSpinBox()
        self.center_right.setRange(-1000, 1000)
        self.center_right.setSuffix(" px")
        self.center_right.setToolTip("Expand the Center crop's right edge outward.")
        tg.addWidget(self.center_right)
        tg.addStretch()
        for sp in (self.left_inner, self.right_inner, self.center_left, self.center_right):
            sp.valueChanged.connect(self._on_thirds_changed)
        self.thirds_group.setVisible(False)
        layout.addWidget(self.thirds_group)

        # Serial overlay crop toggle
        self.overlay_crop_check = QCheckBox(
            "Add serial overlay crop (digit boxes + pattern highlights)")
        self.overlay_crop_check.setToolTip(
            "Append a zoomed serial-number crop with the matched pattern(s) drawn "
            "on it (one per selected pattern; gas-pump targets the shifted serial).")
        layout.addWidget(self.overlay_crop_check)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._save_and_close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Row selection drives the contextual editors (thirds group, seal
        # emphasis) and, when available, the live preview.
        self.crop_table.itemSelectionChanged.connect(self._on_row_selected)

        # Live preview panel (only when a sample bill was supplied)
        if self.preview_ctx:
            outer.addWidget(self._build_preview_panel())
            for spin in (self.front_seal_width, self.front_seal_height,
                         self.front_seal_offset_x, self.front_seal_offset_y,
                         self.back_seal_width, self.back_seal_height,
                         self.back_seal_offset_x, self.back_seal_offset_y):
                spin.valueChanged.connect(self._refresh_preview)

    def _build_preview_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(380)
        v = QVBoxLayout(panel)
        title = QLabel("Preview")
        title.setStyleSheet("font-weight:bold")
        v.addWidget(title)
        self.preview_hint = QLabel("Select a crop to preview it on a sample bill.")
        self.preview_hint.setWordWrap(True)
        self.preview_hint.setStyleSheet("color:#888")
        v.addWidget(self.preview_hint)

        v.addWidget(QLabel("Region on the bill:"))
        self.preview_bill = QLabel()
        self.preview_bill.setMinimumHeight(180)
        self.preview_bill.setAlignment(Qt.AlignCenter)
        self.preview_bill.setFrameShape(QFrame.Box)
        v.addWidget(self.preview_bill)

        v.addWidget(QLabel("Resulting crop:"))
        self.preview_crop = QLabel()
        self.preview_crop.setMinimumHeight(150)
        self.preview_crop.setAlignment(Qt.AlignCenter)
        self.preview_crop.setFrameShape(QFrame.Box)
        v.addWidget(self.preview_crop)

        self.preview_info = QLabel("")
        self.preview_info.setStyleSheet("color:#888")
        v.addWidget(self.preview_info)
        v.addStretch()
        return panel

    def _load_settings(self):
        """Load current settings from config."""
        # Get current crop order from config
        crop_order = self.config.get('crop_order', [])

        # Build ordered list based on config, then add any missing crops
        ordered_crops = []
        seen = set()

        for side, region in crop_order:
            for crop in self.DEFAULT_CROPS:
                if crop['side'] == side and crop['region'] == region:
                    ordered_crops.append(crop.copy())
                    seen.add((side, region))
                    break

        # Add any crops not in the config order
        for crop in self.DEFAULT_CROPS:
            key = (crop['side'], crop['region'])
            if key not in seen:
                ordered_crops.append(crop.copy())

        # Populate table
        self.crop_table.setRowCount(len(ordered_crops))
        for i, crop in enumerate(ordered_crops):
            # Enabled checkbox - enabled if in the current crop_order
            enabled_check = QCheckBox()
            key = (crop['side'], crop['region'])
            is_in_order = key in seen
            enabled_check.setChecked(is_in_order)
            enabled_check.stateChanged.connect(self._update_order_numbers)
            self.crop_table.setCellWidget(i, 0, enabled_check)

            # Crop name
            name_item = QTableWidgetItem(crop['name'])
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            name_item.setData(Qt.UserRole, crop)
            self.crop_table.setItem(i, 1, name_item)

            # Order number (1-based)
            order_item = QTableWidgetItem(f"{i + 1:02d}")
            order_item.setFlags(order_item.flags() & ~Qt.ItemIsEditable)
            order_item.setTextAlignment(Qt.AlignCenter)
            self.crop_table.setItem(i, 2, order_item)

            # Settings indicator
            if crop['has_settings']:
                settings_item = QTableWidgetItem("✓")
                settings_item.setTextAlignment(Qt.AlignCenter)
            else:
                settings_item = QTableWidgetItem("")
            settings_item.setFlags(settings_item.flags() & ~Qt.ItemIsEditable)
            self.crop_table.setItem(i, 3, settings_item)

        # Load seal settings
        yolo_crops = self.config.get('yolo_crops', {})

        # Front seal
        front_seal = yolo_crops.get('front_seal', {})
        self.front_seal_width.setValue(front_seal.get('min_width', 0))
        self.front_seal_height.setValue(front_seal.get('min_height', 0))
        self.front_seal_offset_x.setValue(front_seal.get('offset_x', 0))
        self.front_seal_offset_y.setValue(front_seal.get('offset_y', 0))

        # Back seal
        back_seal = yolo_crops.get('back_seal', {})
        self.back_seal_width.setValue(back_seal.get('width', 500))
        self.back_seal_height.setValue(back_seal.get('height', 500))
        self.back_seal_offset_x.setValue(back_seal.get('offset_x', 0))
        self.back_seal_offset_y.setValue(back_seal.get('offset_y', 0))

        # Serial overlay crop (default on)
        self.overlay_crop_check.setChecked(self.config.get('include_serial_overlay', True))

        # Update order numbers based on enabled state
        self._update_order_numbers()

        # Select the first crop so the contextual editors populate immediately.
        if self.crop_table.rowCount():
            self.crop_table.selectRow(0)
            self._on_row_selected()

    # ------------------------------------------------------------------
    # Contextual editors + live preview
    # ------------------------------------------------------------------
    def _on_row_selected(self):
        self._update_seal_groups()
        self._update_thirds_group()
        self._refresh_preview()

    def _update_thirds_group(self):
        """Show the overlap knobs relevant to the selected left/center/right crop."""
        crop = self._selected_crop()
        region = crop['region'] if crop else None
        is_third = region in ('left', 'center', 'right')
        self.thirds_group.setVisible(is_third)
        if not is_third:
            self._active_thirds = None
            return
        side = crop['side']
        self._active_thirds = (side, region)
        self.thirds_group.setTitle(f"Third-Crop Overlap — {crop['name']}")
        stored = (((self.config.get('yolo_crops') or {}).get('thirds') or {}).get(side) or {})
        # Toggle which knobs are visible for this region
        show_left = region == 'left'
        show_right = region == 'right'
        show_center = region == 'center'
        self.lbl_left_inner.setVisible(show_left); self.left_inner.setVisible(show_left)
        self.lbl_right_inner.setVisible(show_right); self.right_inner.setVisible(show_right)
        self.lbl_center_left.setVisible(show_center); self.center_left.setVisible(show_center)
        self.lbl_center_right.setVisible(show_center); self.center_right.setVisible(show_center)
        # Load stored values without triggering write-back
        for sp, key in ((self.left_inner, 'left_inner'), (self.right_inner, 'right_inner'),
                        (self.center_left, 'center_left'), (self.center_right, 'center_right')):
            sp.blockSignals(True)
            sp.setValue(int(stored.get(key, 0)))
            sp.blockSignals(False)

    def _on_thirds_changed(self):
        if not getattr(self, '_active_thirds', None):
            return
        side, region = self._active_thirds
        node = self.config.setdefault('yolo_crops', {}).setdefault('thirds', {}).setdefault(side, {})
        if region == 'left':
            node['left_inner'] = self.left_inner.value()
        elif region == 'right':
            node['right_inner'] = self.right_inner.value()
        elif region == 'center':
            node['center_left'] = self.center_left.value()
            node['center_right'] = self.center_right.value()
        self._refresh_preview()

    def _selected_crop(self):
        row = self.crop_table.currentRow()
        if row < 0:
            return None
        item = self.crop_table.item(row, 1)
        return item.data(Qt.UserRole) if item else None

    def _update_seal_groups(self):
        """Show a seal's settings only when that seal crop is selected, matching
        the contextual thirds group (avoids implying the seal knobs belong to
        whatever crop is selected)."""
        crop = self._selected_crop()
        side = crop['side'] if crop else None
        is_seal = bool(crop and crop['region'] == 'seal')
        self.front_seal_group.setVisible(is_seal and side == 'front')
        self.back_seal_group.setVisible(is_seal and side == 'back')

    def _current_config_overrides(self):
        import copy
        cfg = copy.deepcopy(self.config) if isinstance(self.config, dict) else {}
        yc = cfg.get('yolo_crops') or {}
        yc.setdefault('front_seal', {})
        yc.setdefault('back_seal', {})
        yc['front_seal']['min_width'] = self.front_seal_width.value()
        yc['front_seal']['min_height'] = self.front_seal_height.value()
        yc['front_seal']['offset_x'] = self.front_seal_offset_x.value()
        yc['front_seal']['offset_y'] = self.front_seal_offset_y.value()
        yc['back_seal']['width'] = self.back_seal_width.value()
        yc['back_seal']['height'] = self.back_seal_height.value()
        yc['back_seal']['offset_x'] = self.back_seal_offset_x.value()
        yc['back_seal']['offset_y'] = self.back_seal_offset_y.value()
        cfg['yolo_crops'] = yc
        return cfg

    def _refresh_preview(self):
        if not self.preview_ctx:
            return
        crop = self._selected_crop()
        if not crop:
            return
        side, region = crop['side'], crop['region']
        if not self.preview_ctx.has_side(side):
            self.preview_hint.setText(f"No {side} sample bill available for preview.")
            self.preview_bill.clear()
            self.preview_crop.clear()
            self.preview_info.clear()
            return
        self.preview_hint.setText(f"{crop['name']}  ({side} / {region})")
        bill, rect, cropimg = self.preview_ctx.render(side, region, self._current_config_overrides())
        if bill is None:
            return
        self._show_bill_with_rect(bill, rect)
        self._show_crop(cropimg)
        if rect:
            x1, y1, x2, y2 = rect
            self.preview_info.setText(f"crop region: {x2 - x1}×{y2 - y1}px  @ ({x1}, {y1})")
        else:
            self.preview_info.setText("(no region for this crop on the sample)")

    def _bgr_to_pixmap(self, bgr, max_w, max_h):
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg).scaled(max_w, max_h, Qt.KeepAspectRatio,
                                              Qt.SmoothTransformation)

    def _show_bill_with_rect(self, bill, rect):
        pm = self._bgr_to_pixmap(bill, 360, 200)
        if rect and pm.width() and pm.height():
            bh, bw = bill.shape[:2]
            sx = pm.width() / bw
            sy = pm.height() / bh
            x1, y1, x2, y2 = rect
            painter = QPainter(pm)
            painter.setPen(QPen(QColor('#00d000'), 3))
            painter.drawRect(int(x1 * sx), int(y1 * sy),
                             int((x2 - x1) * sx), int((y2 - y1) * sy))
            painter.end()
        self.preview_bill.setPixmap(pm)

    def _show_crop(self, cropimg):
        if cropimg is None or cropimg.size == 0:
            self.preview_crop.setText("(empty)")
            return
        self.preview_crop.setPixmap(self._bgr_to_pixmap(cropimg, 360, 150))

    def _update_order_numbers(self):
        """Update the order column to reflect current row positions."""
        order_num = 1
        for i in range(self.crop_table.rowCount()):
            enabled_check = self.crop_table.cellWidget(i, 0)
            if enabled_check and enabled_check.isChecked():
                self.crop_table.item(i, 2).setText(f"{order_num:02d}")
                order_num += 1
            else:
                self.crop_table.item(i, 2).setText("--")

    def _move_up(self):
        """Move selected row up."""
        row = self.crop_table.currentRow()
        if row <= 0:
            return

        self._swap_rows(row, row - 1)
        self.crop_table.selectRow(row - 1)
        self._update_order_numbers()

    def _move_down(self):
        """Move selected row down."""
        row = self.crop_table.currentRow()
        if row < 0 or row >= self.crop_table.rowCount() - 1:
            return

        self._swap_rows(row, row + 1)
        self.crop_table.selectRow(row + 1)
        self._update_order_numbers()

    def _swap_rows(self, row1, row2):
        """Swap two rows in the table."""
        # Swap checkboxes
        check1 = self.crop_table.cellWidget(row1, 0)
        check2 = self.crop_table.cellWidget(row2, 0)
        state1 = check1.isChecked()
        state2 = check2.isChecked()

        # Swap items
        for col in range(1, self.crop_table.columnCount()):
            item1 = self.crop_table.takeItem(row1, col)
            item2 = self.crop_table.takeItem(row2, col)
            self.crop_table.setItem(row1, col, item2)
            self.crop_table.setItem(row2, col, item1)

        # Restore checkbox states (swapped)
        check1.setChecked(state2)
        check2.setChecked(state1)

    def _reset_order(self):
        """Reset to default crop order."""
        reply = QMessageBox.question(
            self, "Reset Order",
            "Reset crop order to defaults?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._load_settings()

    def _save_and_close(self):
        """Save settings to config and close."""
        # Build new crop_order list
        crop_order = []
        for i in range(self.crop_table.rowCount()):
            enabled_check = self.crop_table.cellWidget(i, 0)
            if enabled_check and enabled_check.isChecked():
                crop = self.crop_table.item(i, 1).data(Qt.UserRole)
                crop_order.append([crop['side'], crop['region']])

        self.config['crop_order'] = crop_order
        self.config['include_serial_overlay'] = self.overlay_crop_check.isChecked()

        # Save seal settings
        if 'yolo_crops' not in self.config:
            self.config['yolo_crops'] = {}

        if 'front_seal' not in self.config['yolo_crops']:
            self.config['yolo_crops']['front_seal'] = {}

        self.config['yolo_crops']['front_seal']['min_width'] = self.front_seal_width.value()
        self.config['yolo_crops']['front_seal']['min_height'] = self.front_seal_height.value()
        self.config['yolo_crops']['front_seal']['offset_x'] = self.front_seal_offset_x.value()
        self.config['yolo_crops']['front_seal']['offset_y'] = self.front_seal_offset_y.value()

        if 'back_seal' not in self.config['yolo_crops']:
            self.config['yolo_crops']['back_seal'] = {}

        self.config['yolo_crops']['back_seal']['width'] = self.back_seal_width.value()
        self.config['yolo_crops']['back_seal']['height'] = self.back_seal_height.value()
        self.config['yolo_crops']['back_seal']['offset_x'] = self.back_seal_offset_x.value()
        self.config['yolo_crops']['back_seal']['offset_y'] = self.back_seal_offset_y.value()

        self.accept()

    def get_config(self):
        """Return the modified config."""
        return self.config


# Test dialog standalone
if __name__ == "__main__":
    import yaml
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Load config for testing
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    dialog = EbayCropDialog(config)
    if dialog.exec() == QDialog.Accepted:
        print("Updated config:")
        print(yaml.dump(dialog.get_config(), default_flow_style=False))
