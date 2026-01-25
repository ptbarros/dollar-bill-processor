"""
Settings Dialog - Configure application settings.
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QLineEdit, QPushButton, QDialogButtonBox, QLabel,
    QFileDialog
)
from PySide6.QtCore import Qt

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from settings_manager import SettingsManager


class SettingsDialog(QDialog):
    """Dialog for configuring application settings."""

    def __init__(self, settings: SettingsManager, parent=None):
        super().__init__(parent)
        self.settings = settings

        self.setWindowTitle("Settings")
        self.setMinimumWidth(500)
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)

        # Tab widget
        tabs = QTabWidget()

        # Processing tab
        processing_tab = QWidget()
        self._setup_processing_tab(processing_tab)
        tabs.addTab(processing_tab, "Processing")

        # UI tab
        ui_tab = QWidget()
        self._setup_ui_tab(ui_tab)
        tabs.addTab(ui_tab, "Interface")

        # Export tab
        export_tab = QWidget()
        self._setup_export_tab(export_tab)
        tabs.addTab(export_tab, "Export")

        # Crop tab
        crop_tab = QWidget()
        self._setup_crop_tab(crop_tab)
        tabs.addTab(crop_tab, "Crop Regions")

        layout.addWidget(tabs)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults
        )
        button_box.accepted.connect(self._save_and_accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._restore_defaults)
        layout.addWidget(button_box)

    def _setup_processing_tab(self, tab: QWidget):
        """Setup the processing settings tab."""
        layout = QVBoxLayout(tab)

        # Detection settings
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QFormLayout(detection_group)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setDecimals(2)
        detection_layout.addRow("Confidence threshold:", self.confidence_spin)

        self.multipass_check = QCheckBox("Enable multi-pass detection")
        detection_layout.addRow(self.multipass_check)

        self.max_passes_spin = QSpinBox()
        self.max_passes_spin.setRange(1, 10)
        detection_layout.addRow("Maximum passes:", self.max_passes_spin)

        layout.addWidget(detection_group)

        # Hardware settings
        hardware_group = QGroupBox("Hardware")
        hardware_layout = QFormLayout(hardware_group)

        self.gpu_check = QCheckBox("Use GPU acceleration (if available)")
        hardware_layout.addRow(self.gpu_check)

        self.verify_pairs_check = QCheckBox("Verify front/back pairs")
        hardware_layout.addRow(self.verify_pairs_check)

        layout.addWidget(hardware_group)

        # Output settings
        output_group = QGroupBox("Output")
        output_layout = QFormLayout(output_group)

        self.jpeg_quality_spin = QSpinBox()
        self.jpeg_quality_spin.setRange(50, 100)
        output_layout.addRow("JPEG quality:", self.jpeg_quality_spin)

        self.crop_all_check = QCheckBox("Crop all bills (not just fancy)")
        output_layout.addRow(self.crop_all_check)

        layout.addWidget(output_group)

        layout.addStretch()

    def _setup_ui_tab(self, tab: QWidget):
        """Setup the UI settings tab."""
        layout = QVBoxLayout(tab)

        # Appearance
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QFormLayout(appearance_group)

        self.theme_combo = QComboBox()
        self.theme_combo.addItem("System Default", "system")
        self.theme_combo.addItem("Light", "light")
        self.theme_combo.addItem("Dark", "dark")
        appearance_layout.addRow("Theme:", self.theme_combo)

        self.thumbnails_check = QCheckBox("Show thumbnails in results")
        appearance_layout.addRow(self.thumbnails_check)

        self.thumbnail_size_spin = QSpinBox()
        self.thumbnail_size_spin.setRange(100, 400)
        self.thumbnail_size_spin.setSingleStep(50)
        appearance_layout.addRow("Thumbnail size:", self.thumbnail_size_spin)

        # Font size for accessibility
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 24)
        self.font_size_spin.setSingleStep(1)
        self.font_size_spin.setSuffix(" pt")
        appearance_layout.addRow("Font size:", self.font_size_spin)

        font_hint = QLabel("Larger fonts for easier reading (default: 10 pt)")
        font_hint.setStyleSheet("color: gray; font-size: 9px;")
        appearance_layout.addRow("", font_hint)

        layout.addWidget(appearance_group)

        # Directories
        dirs_group = QGroupBox("Default Directories")
        dirs_layout = QFormLayout(dirs_group)

        self.last_input_edit = QLineEdit()
        self.last_input_edit.setReadOnly(True)
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.last_input_edit)
        input_btn = QPushButton("...")
        input_btn.setMaximumWidth(30)
        input_btn.clicked.connect(self._browse_input)
        input_layout.addWidget(input_btn)
        dirs_layout.addRow("Last input:", input_layout)

        self.last_output_edit = QLineEdit()
        self.last_output_edit.setReadOnly(True)
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.last_output_edit)
        output_btn = QPushButton("...")
        output_btn.setMaximumWidth(30)
        output_btn.clicked.connect(self._browse_output)
        output_layout.addWidget(output_btn)
        dirs_layout.addRow("Last output:", output_layout)

        layout.addWidget(dirs_group)

        layout.addStretch()

    def _setup_export_tab(self, tab: QWidget):
        """Setup the export settings tab."""
        layout = QVBoxLayout(tab)

        # Auto-export options
        auto_group = QGroupBox("Auto-Export After Processing")
        auto_layout = QFormLayout(auto_group)

        self.auto_csv_check = QCheckBox("Automatically generate CSV")
        auto_layout.addRow(self.auto_csv_check)

        self.auto_summary_check = QCheckBox("Automatically generate summary text file")
        auto_layout.addRow(self.auto_summary_check)

        layout.addWidget(auto_group)

        # Default format
        format_group = QGroupBox("Manual Export Format")
        format_layout = QFormLayout(format_group)

        self.format_combo = QComboBox()
        self.format_combo.addItem("CSV", "csv")
        self.format_combo.addItem("Excel", "excel")
        self.format_combo.addItem("HTML Report", "html")
        format_layout.addRow("Format:", self.format_combo)

        self.include_thumbs_check = QCheckBox("Include thumbnails in HTML export")
        format_layout.addRow(self.include_thumbs_check)

        layout.addWidget(format_group)

        # Templates
        templates_group = QGroupBox("Templates (Optional)")
        templates_layout = QFormLayout(templates_group)

        self.excel_template_edit = QLineEdit()
        excel_layout = QHBoxLayout()
        excel_layout.addWidget(self.excel_template_edit)
        excel_btn = QPushButton("...")
        excel_btn.setMaximumWidth(30)
        excel_btn.clicked.connect(self._browse_excel_template)
        excel_layout.addWidget(excel_btn)
        templates_layout.addRow("Excel template:", excel_layout)

        self.html_template_edit = QLineEdit()
        html_layout = QHBoxLayout()
        html_layout.addWidget(self.html_template_edit)
        html_btn = QPushButton("...")
        html_btn.setMaximumWidth(30)
        html_btn.clicked.connect(self._browse_html_template)
        html_layout.addWidget(html_btn)
        templates_layout.addRow("HTML template:", html_layout)

        layout.addWidget(templates_group)

        layout.addStretch()

    def _setup_crop_tab(self, tab: QWidget):
        """Setup the crop region settings tab."""
        layout = QVBoxLayout(tab)

        # Help text
        help_label = QLabel(
            "Crop regions are percentages (0.0-1.0) of image dimensions.\n"
            "X/Y = top-left corner, W/H = width/height of the region."
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        # Front seal crop
        front_group = QGroupBox("Front Seal Crop")
        front_layout = QFormLayout(front_group)

        self.front_seal_x = QDoubleSpinBox()
        self.front_seal_x.setRange(0.0, 1.0)
        self.front_seal_x.setSingleStep(0.01)
        self.front_seal_x.setDecimals(3)
        front_layout.addRow("X (left):", self.front_seal_x)

        self.front_seal_y = QDoubleSpinBox()
        self.front_seal_y.setRange(0.0, 1.0)
        self.front_seal_y.setSingleStep(0.01)
        self.front_seal_y.setDecimals(3)
        front_layout.addRow("Y (top):", self.front_seal_y)

        self.front_seal_w = QDoubleSpinBox()
        self.front_seal_w.setRange(0.0, 1.0)
        self.front_seal_w.setSingleStep(0.01)
        self.front_seal_w.setDecimals(3)
        front_layout.addRow("Width:", self.front_seal_w)

        self.front_seal_h = QDoubleSpinBox()
        self.front_seal_h.setRange(0.0, 1.0)
        self.front_seal_h.setSingleStep(0.01)
        self.front_seal_h.setDecimals(3)
        front_layout.addRow("Height:", self.front_seal_h)

        layout.addWidget(front_group)

        # Back seal crop
        back_group = QGroupBox("Back Seal Crop")
        back_layout = QFormLayout(back_group)

        self.back_seal_x = QDoubleSpinBox()
        self.back_seal_x.setRange(0.0, 1.0)
        self.back_seal_x.setSingleStep(0.01)
        self.back_seal_x.setDecimals(3)
        back_layout.addRow("X (left):", self.back_seal_x)

        self.back_seal_y = QDoubleSpinBox()
        self.back_seal_y.setRange(0.0, 1.0)
        self.back_seal_y.setSingleStep(0.01)
        self.back_seal_y.setDecimals(3)
        back_layout.addRow("Y (top):", self.back_seal_y)

        self.back_seal_w = QDoubleSpinBox()
        self.back_seal_w.setRange(0.0, 1.0)
        self.back_seal_w.setSingleStep(0.01)
        self.back_seal_w.setDecimals(3)
        back_layout.addRow("Width:", self.back_seal_w)

        self.back_seal_h = QDoubleSpinBox()
        self.back_seal_h.setRange(0.0, 1.0)
        self.back_seal_h.setSingleStep(0.01)
        self.back_seal_h.setDecimals(3)
        back_layout.addRow("Height:", self.back_seal_h)

        layout.addWidget(back_group)

        layout.addStretch()

    def _load_settings(self):
        """Load current settings into the UI."""
        # Processing
        self.confidence_spin.setValue(self.settings.processing.confidence_threshold)
        self.multipass_check.setChecked(self.settings.processing.multi_pass_detection)
        self.max_passes_spin.setValue(self.settings.processing.max_detection_passes)
        self.gpu_check.setChecked(self.settings.processing.use_gpu)
        self.verify_pairs_check.setChecked(self.settings.processing.verify_pairs)
        self.jpeg_quality_spin.setValue(self.settings.processing.jpeg_quality)
        self.crop_all_check.setChecked(self.settings.processing.crop_all)

        # UI
        idx = self.theme_combo.findData(self.settings.ui.theme)
        if idx >= 0:
            self.theme_combo.setCurrentIndex(idx)
        self.thumbnails_check.setChecked(self.settings.ui.show_thumbnails)
        self.thumbnail_size_spin.setValue(self.settings.ui.thumbnail_size)
        self.font_size_spin.setValue(self.settings.ui.font_size)
        self.last_input_edit.setText(self.settings.ui.last_input_dir)
        self.last_output_edit.setText(self.settings.ui.last_output_dir)

        # Export
        idx = self.format_combo.findData(self.settings.export.default_format)
        if idx >= 0:
            self.format_combo.setCurrentIndex(idx)
        self.include_thumbs_check.setChecked(self.settings.export.include_thumbnails)
        self.excel_template_edit.setText(self.settings.export.excel_template)
        self.html_template_edit.setText(self.settings.export.html_template)
        self.auto_csv_check.setChecked(self.settings.export.auto_export_csv)
        self.auto_summary_check.setChecked(self.settings.export.auto_export_summary)

        # Crop
        self.front_seal_x.setValue(self.settings.crop.front_seal_x)
        self.front_seal_y.setValue(self.settings.crop.front_seal_y)
        self.front_seal_w.setValue(self.settings.crop.front_seal_w)
        self.front_seal_h.setValue(self.settings.crop.front_seal_h)
        self.back_seal_x.setValue(self.settings.crop.back_seal_x)
        self.back_seal_y.setValue(self.settings.crop.back_seal_y)
        self.back_seal_w.setValue(self.settings.crop.back_seal_w)
        self.back_seal_h.setValue(self.settings.crop.back_seal_h)

    def _save_settings(self):
        """Save UI values to settings."""
        # Processing
        self.settings.processing.confidence_threshold = self.confidence_spin.value()
        self.settings.processing.multi_pass_detection = self.multipass_check.isChecked()
        self.settings.processing.max_detection_passes = self.max_passes_spin.value()
        self.settings.processing.use_gpu = self.gpu_check.isChecked()
        self.settings.processing.verify_pairs = self.verify_pairs_check.isChecked()
        self.settings.processing.jpeg_quality = self.jpeg_quality_spin.value()
        self.settings.processing.crop_all = self.crop_all_check.isChecked()

        # UI
        self.settings.ui.theme = self.theme_combo.currentData()
        self.settings.ui.show_thumbnails = self.thumbnails_check.isChecked()
        self.settings.ui.thumbnail_size = self.thumbnail_size_spin.value()
        self.settings.ui.font_size = self.font_size_spin.value()
        self.settings.ui.last_input_dir = self.last_input_edit.text()
        self.settings.ui.last_output_dir = self.last_output_edit.text()

        # Export
        self.settings.export.default_format = self.format_combo.currentData()
        self.settings.export.include_thumbnails = self.include_thumbs_check.isChecked()
        self.settings.export.excel_template = self.excel_template_edit.text()
        self.settings.export.html_template = self.html_template_edit.text()
        self.settings.export.auto_export_csv = self.auto_csv_check.isChecked()
        self.settings.export.auto_export_summary = self.auto_summary_check.isChecked()

        # Crop
        self.settings.crop.front_seal_x = self.front_seal_x.value()
        self.settings.crop.front_seal_y = self.front_seal_y.value()
        self.settings.crop.front_seal_w = self.front_seal_w.value()
        self.settings.crop.front_seal_h = self.front_seal_h.value()
        self.settings.crop.back_seal_x = self.back_seal_x.value()
        self.settings.crop.back_seal_y = self.back_seal_y.value()
        self.settings.crop.back_seal_w = self.back_seal_w.value()
        self.settings.crop.back_seal_h = self.back_seal_h.value()

    def _save_and_accept(self):
        """Save settings and close."""
        self._save_settings()
        self.accept()

    def _restore_defaults(self):
        """Restore default settings."""
        from settings_manager import ProcessingSettings, UISettings, ExportSettings, CropSettings

        # Reset to defaults
        self.settings.processing = ProcessingSettings()
        self.settings.ui = UISettings()
        self.settings.export = ExportSettings()
        self.settings.crop = CropSettings()

        # Reload UI
        self._load_settings()

    def _browse_input(self):
        """Browse for input directory."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Default Input Directory",
            self.last_input_edit.text() or str(Path.home())
        )
        if folder:
            self.last_input_edit.setText(folder)

    def _browse_output(self):
        """Browse for output directory."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Default Output Directory",
            self.last_output_edit.text() or str(Path.home())
        )
        if folder:
            self.last_output_edit.setText(folder)

    def _browse_excel_template(self):
        """Browse for Excel template."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Excel Template",
            str(Path.home()),
            "Excel Files (*.xlsx)"
        )
        if path:
            self.excel_template_edit.setText(path)

    def _browse_html_template(self):
        """Browse for HTML template."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select HTML Template",
            str(Path.home()),
            "HTML Files (*.html)"
        )
        if path:
            self.html_template_edit.setText(path)
