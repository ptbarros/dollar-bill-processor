"""
Settings Dialog - Configure application settings.
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QLineEdit, QPushButton, QDialogButtonBox, QLabel,
    QFileDialog, QColorDialog, QMessageBox
)
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtCore import Qt, QUrl


class ClickableInfoLabel(QLabel):
    """Info label that shows tooltip on hover."""

    def __init__(self, text: str, tooltip: str, parent=None):
        super().__init__(text, parent)
        self.setToolTip(tooltip)
        self.setStyleSheet("color: #888; font-size: 14px;")
        self.setCursor(Qt.WhatsThisCursor)

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from settings_manager import SettingsManager


class SettingsDialog(QDialog):
    """Dialog for configuring application settings."""

    def __init__(self, settings: SettingsManager, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._fancy_color = settings.ui.default_fancy_color or "#2e7d32"

        self.setWindowTitle("Settings")
        self.setMinimumWidth(500)
        self._setup_ui()
        self._load_settings()

    def _create_checkbox_with_info(self, label: str, tooltip: str) -> tuple[QCheckBox, QWidget]:
        """Create a checkbox with an info icon that shows a tooltip.

        Returns:
            Tuple of (checkbox, container_widget) - use container for layout, checkbox for state
        """
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        checkbox = QCheckBox(label)
        layout.addWidget(checkbox)

        info_label = ClickableInfoLabel("ⓘ", tooltip)
        layout.addWidget(info_label)

        layout.addStretch()

        return checkbox, container

    def _create_widget_with_info(self, widget: QWidget, tooltip: str) -> QWidget:
        """Wrap any widget with an info icon next to it.

        Returns:
            Container widget with original widget + info icon
        """
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(widget)
        layout.addWidget(ClickableInfoLabel("ⓘ", tooltip))
        layout.addStretch()

        return container

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

        # Monitor tab
        monitor_tab = QWidget()
        self._setup_monitor_tab(monitor_tab)
        tabs.addTab(monitor_tab, "Monitor")

        # AI tab
        ai_tab = QWidget()
        self._setup_ai_tab(ai_tab)
        tabs.addTab(ai_tab, "AI")

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
        confidence_container = self._create_widget_with_info(
            self.confidence_spin,
            "Minimum confidence level for OCR results.\n\n"
            "• Higher (0.7-0.9): More accurate, but may miss some serials\n"
            "• Lower (0.3-0.5): Catches more serials, but more errors\n\n"
            "Bills below this threshold go to the review queue.\n"
            "Default: 0.5 (balanced)"
        )
        detection_layout.addRow("Confidence threshold:", confidence_container)

        self.multipass_check, multipass_container = self._create_checkbox_with_info(
            "Enable multi-pass detection",
            "Try multiple detection strategies if the first pass fails.\n\n"
            "• Checked: Retries with different settings (slower but thorough)\n"
            "• Unchecked: Single pass only (faster)\n\n"
            "Useful for poor quality scans or unusual bill orientations."
        )
        detection_layout.addRow(multipass_container)

        self.max_passes_spin = QSpinBox()
        self.max_passes_spin.setRange(1, 10)
        max_passes_container = self._create_widget_with_info(
            self.max_passes_spin,
            "Maximum detection attempts when multi-pass is enabled.\n\n"
            "• 2-3: Quick retry for minor issues\n"
            "• 5-7: Thorough for difficult scans\n\n"
            "Higher values = slower but may recover more serials."
        )
        detection_layout.addRow("Maximum passes:", max_passes_container)

        layout.addWidget(detection_group)

        # Hardware settings
        hardware_group = QGroupBox("Hardware")
        hardware_layout = QFormLayout(hardware_group)

        self.gpu_check, gpu_container = self._create_checkbox_with_info(
            "Use GPU acceleration (DirectML / CUDA if available)",
            "Let ONNX Runtime run YOLO detection on the GPU.\n\n"
            "• Checked: auto-selects the best provider (DirectML on Windows,\n"
            "  CUDA where available), falling back to CPU if there's no GPU.\n"
            "• Unchecked: forces CPU only.\n\n"
            "Uncheck to compare CPU vs GPU speed in the same build (see the\n"
            "Rate line in the debug log)."
        )
        hardware_layout.addRow(gpu_container)

        self.debug_log_check, debug_container = self._create_checkbox_with_info(
            "Write processing debug log",
            "Write per-bill timing and the batch summary (with backend/provider\n"
            "and bills-per-minute) to debug_log.txt.\n\n"
            "Use this to compare processing speed between runs and machines.\n"
            "Turn it off for normal use to keep the log small."
        )
        hardware_layout.addRow(debug_container)

        view_log_btn = QPushButton("View debug log…")
        view_log_btn.clicked.connect(self._open_debug_log)
        hardware_layout.addRow("", view_log_btn)

        self.verify_pairs_check, verify_container = self._create_checkbox_with_info(
            "Verify front/back pairs",
            "How to detect which image is the front (serial side) of each bill.\n\n"
            "• Checked: Verify all pairs upfront before processing.\n"
            "  Slower startup, but consistent processing speed.\n"
            "  Best for: Unsorted piles with random orientations.\n\n"
            "• Unchecked: Lazy detection during processing.\n"
            "  Fast startup, swaps on-demand if no serial found.\n"
            "  Best for: Scanner output where pairs are usually correct."
        )
        hardware_layout.addRow(verify_container)

        layout.addWidget(hardware_group)

        # Output settings
        output_group = QGroupBox("Output")
        output_layout = QFormLayout(output_group)

        self.jpeg_quality_spin = QSpinBox()
        self.jpeg_quality_spin.setRange(50, 100)
        output_layout.addRow("JPEG quality:", self.jpeg_quality_spin)

        self.crop_all_check, crop_all_container = self._create_checkbox_with_info(
            "Crop all bills (not just fancy)",
            "Generate cropped images for every bill, not just fancy ones.\n\n"
            "• Checked: Save crops for ALL bills (uses more disk space)\n"
            "• Unchecked: Only save crops for bills with fancy patterns\n\n"
            "Useful if you want to archive or manually review all serials."
        )
        output_layout.addRow(crop_all_container)

        self.auto_crop_check, auto_crop_container = self._create_checkbox_with_info(
            "Auto-crop during processing",
            "Automatically generate crops for fancy bills during processing.\n\n"
            "• Checked: Crops generated automatically (default behavior)\n"
            "• Unchecked: No auto-cropping; use manual Crop button/menu\n\n"
            "Disable this to review bills first, then crop selected ones."
        )
        output_layout.addRow(auto_crop_container)

        self.proc_auto_archive_check, auto_archive_container = self._create_checkbox_with_info(
            "Archive after processing",
            "Move processed files to a timestamped archive folder.\n\n"
            "• Checked: Files moved to archive/batch_YYYYMMDD_HHMMSS/\n"
            "• Unchecked: Files stay in original location\n\n"
            "Uses the archive directory from Monitor settings."
        )
        output_layout.addRow(auto_archive_container)

        self.archive_copy_mode_check, copy_mode_container = self._create_checkbox_with_info(
            "Copy instead of move (testing mode)",
            "Copy files to archive instead of moving them.\n\n"
            "• Checked: Original files remain in place (for testing)\n"
            "• Unchecked: Files are moved to archive (normal behavior)\n\n"
            "Useful when testing to avoid manually copying files back."
        )
        output_layout.addRow(copy_mode_container)

        self.extract_plate_info_check, plate_info_container = self._create_checkbox_with_info(
            "Extract plate and series info",
            "Extract additional info from bills: series year, front plate, back plate.\n\n"
            "• Checked: Runs additional OCR on detected regions (slower)\n"
            "• Unchecked: Only extracts serial number (faster)\n\n"
            "Useful for detailed record keeping or research."
        )
        output_layout.addRow(plate_info_container)

        layout.addWidget(output_group)

        # Autosave settings
        autosave_group = QGroupBox("Session Recovery")
        autosave_layout = QFormLayout(autosave_group)

        self.autosave_check, autosave_container = self._create_checkbox_with_info(
            "Enable autosave",
            "Periodically save session state for crash recovery.\n\n"
            "If the app crashes or power is lost, you can restore\n"
            "your progress when you restart.\n\n"
            "Recovery file: .session_recovery.json in app folder"
        )
        autosave_layout.addRow(autosave_container)

        self.autosave_interval_spin = QSpinBox()
        self.autosave_interval_spin.setRange(10, 300)
        self.autosave_interval_spin.setSingleStep(10)
        self.autosave_interval_spin.setSuffix(" seconds")
        interval_container = self._create_widget_with_info(
            self.autosave_interval_spin,
            "How often to save session state.\n\n"
            "• 10-30s: Frequent saves, minimal data loss\n"
            "• 60-120s: Balanced performance/protection\n"
            "• 180-300s: Less frequent, lower overhead\n\n"
            "Default: 30 seconds"
        )
        autosave_layout.addRow("Save interval:", interval_container)

        layout.addWidget(autosave_group)

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

        # Font size for accessibility
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 24)
        self.font_size_spin.setSingleStep(1)
        self.font_size_spin.setSuffix(" pt")
        appearance_layout.addRow("Font size:", self.font_size_spin)

        font_hint = QLabel("Larger fonts for easier reading (default: 10 pt)")
        font_hint.setStyleSheet("color: gray; font-size: 9px;")
        appearance_layout.addRow("", font_hint)

        # Default fancy color
        fancy_color_layout = QHBoxLayout()
        self.fancy_color_btn = QPushButton()
        self.fancy_color_btn.setMinimumWidth(80)
        self.fancy_color_btn.setMaximumWidth(80)
        self.fancy_color_btn.clicked.connect(self._pick_fancy_color)
        fancy_color_layout.addWidget(self.fancy_color_btn)
        fancy_color_layout.addStretch()
        appearance_layout.addRow("Default Fancy Color:", fancy_color_layout)

        fancy_color_hint = QLabel("Color for fancy bills without pattern-specific colors")
        fancy_color_hint.setStyleSheet("color: gray; font-size: 9px;")
        appearance_layout.addRow("", fancy_color_hint)

        layout.addWidget(appearance_group)

        # Default Working Directory
        dirs_group = QGroupBox("Default Working Directory")
        dirs_layout = QFormLayout(dirs_group)

        self.working_dir_edit = QLineEdit()
        self.working_dir_edit.setPlaceholderText("Starting directory for file browse dialogs...")
        working_layout = QHBoxLayout()
        working_layout.addWidget(self.working_dir_edit)
        working_btn = QPushButton("...")
        working_btn.setMaximumWidth(30)
        working_btn.clicked.connect(self._browse_working_dir)
        working_layout.addWidget(working_btn)
        dirs_layout.addRow("Directory:", working_layout)

        working_hint = QLabel("Browse dialogs will start here instead of your home folder")
        working_hint.setStyleSheet("color: gray; font-size: 9px;")
        dirs_layout.addRow("", working_hint)

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

    def _setup_monitor_tab(self, tab: QWidget):
        """Setup the monitor mode settings tab."""
        layout = QVBoxLayout(tab)

        # Directories
        dirs_group = QGroupBox("Monitor Mode Directories")
        dirs_layout = QFormLayout(dirs_group)

        # Watch directory
        self.watch_dir_edit = QLineEdit()
        self.watch_dir_edit.setPlaceholderText("Directory where scanner saves files...")
        watch_layout = QHBoxLayout()
        watch_layout.addWidget(self.watch_dir_edit)
        watch_btn = QPushButton("...")
        watch_btn.setMaximumWidth(30)
        watch_btn.clicked.connect(self._browse_watch_dir)
        watch_layout.addWidget(watch_btn)
        dirs_layout.addRow("Watch Directory:", watch_layout)

        watch_hint = QLabel("Scanner saves files here - monitored for new images")
        watch_hint.setStyleSheet("color: gray; font-size: 9px;")
        dirs_layout.addRow("", watch_hint)

        # Output directory
        self.monitor_output_edit = QLineEdit()
        self.monitor_output_edit.setPlaceholderText("Directory for fancy bill crops...")
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.monitor_output_edit)
        output_btn = QPushButton("...")
        output_btn.setMaximumWidth(30)
        output_btn.clicked.connect(self._browse_monitor_output)
        output_layout.addWidget(output_btn)
        dirs_layout.addRow("Output Directory:", output_layout)

        # Archive directory
        self.archive_dir_edit = QLineEdit()
        self.archive_dir_edit.setPlaceholderText("Directory for completed batches...")
        archive_layout = QHBoxLayout()
        archive_layout.addWidget(self.archive_dir_edit)
        archive_btn = QPushButton("...")
        archive_btn.setMaximumWidth(30)
        archive_btn.clicked.connect(self._browse_archive_dir)
        archive_layout.addWidget(archive_btn)
        dirs_layout.addRow("Archive Directory:", archive_layout)

        archive_hint = QLabel("Processed files are moved here when monitoring stops")
        archive_hint.setStyleSheet("color: gray; font-size: 9px;")
        dirs_layout.addRow("", archive_hint)

        # Review directory
        self.review_dir_edit = QLineEdit()
        self.review_dir_edit.setPlaceholderText("Where 'Save for Review' copies bills...")
        review_layout = QHBoxLayout()
        review_layout.addWidget(self.review_dir_edit)
        review_btn = QPushButton("...")
        review_btn.setMaximumWidth(30)
        review_btn.clicked.connect(self._browse_review_dir)
        review_layout.addWidget(review_btn)
        dirs_layout.addRow("Review Directory:", review_layout)

        review_hint = QLabel("Blank = default per-user location")
        review_hint.setStyleSheet("color: gray; font-size: 9px;")
        dirs_layout.addRow("", review_hint)

        layout.addWidget(dirs_group)

        # Options
        options_group = QGroupBox("Monitor Options")
        options_layout = QFormLayout(options_group)

        self.mon_auto_archive_check = QCheckBox("Auto-archive on stop")
        self.mon_auto_archive_check.setToolTip("Move processed files to timestamped directory when monitoring stops")
        options_layout.addRow(self.mon_auto_archive_check)

        self.poll_interval_spin = QDoubleSpinBox()
        self.poll_interval_spin.setRange(0.1, 10.0)
        self.poll_interval_spin.setSingleStep(0.1)
        self.poll_interval_spin.setDecimals(1)
        self.poll_interval_spin.setSuffix(" seconds")
        options_layout.addRow("Poll Interval:", self.poll_interval_spin)

        poll_hint = QLabel("How often to check for new files (0.5s recommended)")
        poll_hint.setStyleSheet("color: gray; font-size: 9px;")
        options_layout.addRow("", poll_hint)

        self.settle_time_spin = QDoubleSpinBox()
        self.settle_time_spin.setRange(0.1, 5.0)
        self.settle_time_spin.setSingleStep(0.1)
        self.settle_time_spin.setDecimals(1)
        self.settle_time_spin.setSuffix(" seconds")
        options_layout.addRow("File Settle Time:", self.settle_time_spin)

        settle_hint = QLabel("Wait for file to finish writing before processing")
        settle_hint.setStyleSheet("color: gray; font-size: 9px;")
        options_layout.addRow("", settle_hint)

        layout.addWidget(options_group)

        layout.addStretch()

    def _setup_ai_tab(self, tab: QWidget):
        """Setup the AI-assisted pattern generation settings tab."""
        layout = QVBoxLayout(tab)

        # Provider settings
        provider_group = QGroupBox("AI Provider")
        provider_layout = QFormLayout(provider_group)

        self.ai_provider_combo = QComboBox()
        self.ai_provider_combo.addItem("(Not configured)", "")
        self.ai_provider_combo.addItem("Anthropic (Claude)", "anthropic")
        self.ai_provider_combo.addItem("OpenAI (GPT)", "openai")
        self.ai_provider_combo.currentIndexChanged.connect(self._on_ai_provider_changed)
        provider_layout.addRow("Provider:", self.ai_provider_combo)

        layout.addWidget(provider_group)

        # API Keys - separate fields for each provider
        keys_group = QGroupBox("API Keys")
        keys_layout = QFormLayout(keys_group)

        # Anthropic API Key
        self.anthropic_api_key_edit = QLineEdit()
        self.anthropic_api_key_edit.setEchoMode(QLineEdit.Password)
        self.anthropic_api_key_edit.setPlaceholderText("sk-ant-...")

        anthropic_key_layout = QHBoxLayout()
        anthropic_key_layout.addWidget(self.anthropic_api_key_edit)

        self.show_anthropic_key_btn = QPushButton("Show")
        self.show_anthropic_key_btn.setMaximumWidth(50)
        self.show_anthropic_key_btn.setCheckable(True)
        self.show_anthropic_key_btn.toggled.connect(
            lambda checked: self._toggle_key_visibility(self.anthropic_api_key_edit, self.show_anthropic_key_btn, checked))
        anthropic_key_layout.addWidget(self.show_anthropic_key_btn)

        keys_layout.addRow("Anthropic:", anthropic_key_layout)

        # OpenAI API Key
        self.openai_api_key_edit = QLineEdit()
        self.openai_api_key_edit.setEchoMode(QLineEdit.Password)
        self.openai_api_key_edit.setPlaceholderText("sk-...")

        openai_key_layout = QHBoxLayout()
        openai_key_layout.addWidget(self.openai_api_key_edit)

        self.show_openai_key_btn = QPushButton("Show")
        self.show_openai_key_btn.setMaximumWidth(50)
        self.show_openai_key_btn.setCheckable(True)
        self.show_openai_key_btn.toggled.connect(
            lambda checked: self._toggle_key_visibility(self.openai_api_key_edit, self.show_openai_key_btn, checked))
        openai_key_layout.addWidget(self.show_openai_key_btn)

        keys_layout.addRow("OpenAI:", openai_key_layout)

        key_hint = QLabel("API keys are stored locally in user_settings.yaml")
        key_hint.setStyleSheet("color: gray; font-size: 9px;")
        keys_layout.addRow("", key_hint)

        layout.addWidget(keys_group)

        # Model settings
        model_group = QGroupBox("Model Selection")
        model_layout = QFormLayout(model_group)

        # Anthropic model
        self.anthropic_model_combo = QComboBox()
        self.anthropic_model_combo.addItems([
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
        ])
        self.anthropic_model_combo.setEditable(True)  # Allow custom model names
        model_layout.addRow("Anthropic Model:", self.anthropic_model_combo)

        # OpenAI model
        self.openai_model_combo = QComboBox()
        self.openai_model_combo.addItems([
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "o4-mini",
        ])
        self.openai_model_combo.setEditable(True)  # Allow custom model names
        model_layout.addRow("OpenAI Model:", self.openai_model_combo)

        layout.addWidget(model_group)

        # Test connection
        test_group = QGroupBox("Test Connection")
        test_layout = QVBoxLayout(test_group)

        test_btn_layout = QHBoxLayout()
        self.test_ai_btn = QPushButton("Test Connection")
        self.test_ai_btn.clicked.connect(self._test_ai_connection)
        test_btn_layout.addWidget(self.test_ai_btn)
        test_btn_layout.addStretch()
        test_layout.addLayout(test_btn_layout)

        self.ai_test_result = QLabel("")
        self.ai_test_result.setWordWrap(True)
        test_layout.addWidget(self.ai_test_result)

        layout.addWidget(test_group)

        layout.addStretch()

    def _on_ai_provider_changed(self, index: int):
        """Handle AI provider selection change."""
        provider = self.ai_provider_combo.currentData()
        # Could enable/disable relevant model combo based on provider
        # For now, just clear test result
        self.ai_test_result.setText("")

    def _toggle_key_visibility(self, edit: QLineEdit, btn: QPushButton, checked: bool):
        """Toggle API key visibility for a specific field."""
        if checked:
            edit.setEchoMode(QLineEdit.Normal)
            btn.setText("Hide")
        else:
            edit.setEchoMode(QLineEdit.Password)
            btn.setText("Show")

    def _test_ai_connection(self):
        """Test the AI API connection."""
        provider = self.ai_provider_combo.currentData()

        if not provider:
            self.ai_test_result.setText("Please select a provider first.")
            self.ai_test_result.setStyleSheet("color: orange;")
            return

        # Get the API key for the selected provider
        if provider == "anthropic":
            api_key = self.anthropic_api_key_edit.text().strip()
        elif provider == "openai":
            api_key = self.openai_api_key_edit.text().strip()
        else:
            api_key = ""

        if not api_key:
            self.ai_test_result.setText(f"Please enter an API key for {provider}.")
            self.ai_test_result.setStyleSheet("color: orange;")
            return

        self.ai_test_result.setText("Testing connection...")
        self.ai_test_result.setStyleSheet("color: gray;")

        # Force UI update
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            if provider == "anthropic":
                model = self.anthropic_model_combo.currentText()
                self._test_anthropic(api_key, model)
            elif provider == "openai":
                model = self.openai_model_combo.currentText()
                self._test_openai(api_key, model)
        except Exception as e:
            self.ai_test_result.setText(f"Error: {str(e)}")
            self.ai_test_result.setStyleSheet("color: red;")

    def _test_anthropic(self, api_key: str, model: str):
        """Test Anthropic API connection."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Say 'OK'"}]
            )
            self.ai_test_result.setText(f"✓ Connected to Anthropic ({model})")
            self.ai_test_result.setStyleSheet("color: green;")
        except ImportError:
            self.ai_test_result.setText("Error: 'anthropic' package not installed.\nRun: pip install anthropic")
            self.ai_test_result.setStyleSheet("color: red;")
        except Exception as e:
            self.ai_test_result.setText(f"Error: {str(e)}")
            self.ai_test_result.setStyleSheet("color: red;")

    def _test_openai(self, api_key: str, model: str):
        """Test OpenAI API connection."""
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Say 'OK'"}]
            )
            self.ai_test_result.setText(f"✓ Connected to OpenAI ({model})")
            self.ai_test_result.setStyleSheet("color: green;")
        except ImportError:
            self.ai_test_result.setText("Error: 'openai' package not installed.\nRun: pip install openai")
            self.ai_test_result.setStyleSheet("color: red;")
        except Exception as e:
            self.ai_test_result.setText(f"Error: {str(e)}")
            self.ai_test_result.setStyleSheet("color: red;")

    def _open_debug_log(self):
        """Open debug_log.txt in the OS default text viewer."""
        from debug_logger import get_log_path
        path = get_log_path()
        if path and Path(path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        else:
            QMessageBox.information(
                self, "Debug Log",
                "No debug log yet.\n\nEnable \"Write processing debug log\" and "
                "run a batch first."
            )

    def _load_settings(self):
        """Load current settings into the UI."""
        # Processing
        self.confidence_spin.setValue(self.settings.processing.confidence_threshold)
        self.multipass_check.setChecked(self.settings.processing.multi_pass_detection)
        self.max_passes_spin.setValue(self.settings.processing.max_detection_passes)
        self.gpu_check.setChecked(self.settings.processing.gpu_acceleration)
        self.debug_log_check.setChecked(self.settings.processing.debug_logging)
        self.verify_pairs_check.setChecked(self.settings.processing.verify_pairs)
        self.jpeg_quality_spin.setValue(self.settings.processing.jpeg_quality)
        self.crop_all_check.setChecked(self.settings.processing.crop_all)
        self.auto_crop_check.setChecked(self.settings.processing.auto_crop)
        self.proc_auto_archive_check.setChecked(self.settings.processing.auto_archive)
        self.archive_copy_mode_check.setChecked(self.settings.processing.archive_copy_mode)
        self.extract_plate_info_check.setChecked(self.settings.processing.extract_plate_info)

        # Autosave
        self.autosave_check.setChecked(self.settings.autosave.enabled)
        self.autosave_interval_spin.setValue(self.settings.autosave.interval_seconds)

        # UI
        idx = self.theme_combo.findData(self.settings.ui.theme)
        if idx >= 0:
            self.theme_combo.setCurrentIndex(idx)
        self.font_size_spin.setValue(self.settings.ui.font_size)
        self._fancy_color = self.settings.ui.default_fancy_color or "#2e7d32"
        self._update_fancy_color_button()
        self.working_dir_edit.setText(self.settings.ui.default_working_dir)

        # Export
        idx = self.format_combo.findData(self.settings.export.default_format)
        if idx >= 0:
            self.format_combo.setCurrentIndex(idx)
        self.excel_template_edit.setText(self.settings.export.excel_template)
        self.html_template_edit.setText(self.settings.export.html_template)
        self.auto_csv_check.setChecked(self.settings.export.auto_export_csv)
        self.auto_summary_check.setChecked(self.settings.export.auto_export_summary)

        # Monitor
        self.watch_dir_edit.setText(self.settings.monitor.watch_directory)
        self.monitor_output_edit.setText(self.settings.monitor.output_directory)
        self.archive_dir_edit.setText(self.settings.monitor.archive_directory)
        self.review_dir_edit.setText(self.settings.ui.review_directory)
        self.mon_auto_archive_check.setChecked(self.settings.monitor.auto_archive)
        self.poll_interval_spin.setValue(self.settings.monitor.poll_interval)
        self.settle_time_spin.setValue(self.settings.monitor.file_settle_time)

        # AI
        provider = self.settings.ai.provider
        idx = self.ai_provider_combo.findData(provider)
        if idx >= 0:
            self.ai_provider_combo.setCurrentIndex(idx)
        self.anthropic_api_key_edit.setText(self.settings.ai.anthropic_api_key)
        self.openai_api_key_edit.setText(self.settings.ai.openai_api_key)
        self.anthropic_model_combo.setCurrentText(self.settings.ai.anthropic_model)
        self.openai_model_combo.setCurrentText(self.settings.ai.openai_model)

    def _save_settings(self):
        """Save UI values to settings."""
        # Processing
        self.settings.processing.confidence_threshold = self.confidence_spin.value()
        self.settings.processing.multi_pass_detection = self.multipass_check.isChecked()
        self.settings.processing.max_detection_passes = self.max_passes_spin.value()
        self.settings.processing.gpu_acceleration = self.gpu_check.isChecked()
        self.settings.processing.debug_logging = self.debug_log_check.isChecked()
        self.settings.processing.verify_pairs = self.verify_pairs_check.isChecked()
        self.settings.processing.jpeg_quality = self.jpeg_quality_spin.value()
        self.settings.processing.crop_all = self.crop_all_check.isChecked()
        self.settings.processing.auto_crop = self.auto_crop_check.isChecked()
        self.settings.processing.auto_archive = self.proc_auto_archive_check.isChecked()
        self.settings.processing.archive_copy_mode = self.archive_copy_mode_check.isChecked()
        self.settings.processing.extract_plate_info = self.extract_plate_info_check.isChecked()

        # Autosave
        self.settings.autosave.enabled = self.autosave_check.isChecked()
        self.settings.autosave.interval_seconds = self.autosave_interval_spin.value()

        # UI
        self.settings.ui.theme = self.theme_combo.currentData()
        self.settings.ui.font_size = self.font_size_spin.value()
        self.settings.ui.default_fancy_color = self._fancy_color
        self.settings.ui.default_working_dir = self.working_dir_edit.text()

        # Export
        self.settings.export.default_format = self.format_combo.currentData()
        self.settings.export.excel_template = self.excel_template_edit.text()
        self.settings.export.html_template = self.html_template_edit.text()
        self.settings.export.auto_export_csv = self.auto_csv_check.isChecked()
        self.settings.export.auto_export_summary = self.auto_summary_check.isChecked()

        # Monitor
        self.settings.monitor.watch_directory = self.watch_dir_edit.text()
        self.settings.monitor.output_directory = self.monitor_output_edit.text()
        self.settings.monitor.archive_directory = self.archive_dir_edit.text()
        self.settings.ui.review_directory = self.review_dir_edit.text().strip()
        self.settings.monitor.auto_archive = self.mon_auto_archive_check.isChecked()
        self.settings.monitor.poll_interval = self.poll_interval_spin.value()
        self.settings.monitor.file_settle_time = self.settle_time_spin.value()

        # AI
        self.settings.ai.provider = self.ai_provider_combo.currentData() or ""
        self.settings.ai.anthropic_api_key = self.anthropic_api_key_edit.text()
        self.settings.ai.openai_api_key = self.openai_api_key_edit.text()
        self.settings.ai.anthropic_model = self.anthropic_model_combo.currentText()
        self.settings.ai.openai_model = self.openai_model_combo.currentText()

    def _save_and_accept(self):
        """Save settings and close."""
        self._save_settings()
        self.accept()

    def _restore_defaults(self):
        """Restore default settings."""
        from settings_manager import ProcessingSettings, UISettings, ExportSettings, MonitorSettings, AutosaveSettings, AISettings

        # Reset to defaults
        self.settings.processing = ProcessingSettings()
        self.settings.ui = UISettings()
        self.settings.export = ExportSettings()
        self.settings.monitor = MonitorSettings()
        self.settings.autosave = AutosaveSettings()
        self.settings.ai = AISettings()

        # Reset fancy color
        self._fancy_color = "#2e7d32"

        # Reload UI
        self._load_settings()

    def _browse_working_dir(self):
        """Browse for default working directory."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Default Working Directory",
            self.working_dir_edit.text() or str(Path.home())
        )
        if folder:
            self.working_dir_edit.setText(folder)

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

    def _pick_fancy_color(self):
        """Open color picker for default fancy color."""
        current_color = QColor(self._fancy_color)
        color = QColorDialog.getColor(current_color, self, "Select Default Fancy Color")
        if color.isValid():
            self._fancy_color = color.name()
            self._update_fancy_color_button()

    def _update_fancy_color_button(self):
        """Update the fancy color button's appearance."""
        color = QColor(self._fancy_color)
        # Calculate text color based on brightness
        brightness = (color.red() * 299 + color.green() * 587 + color.blue() * 114) / 1000
        text_color = "#000000" if brightness > 128 else "#ffffff"
        self.fancy_color_btn.setStyleSheet(
            f"background-color: {self._fancy_color}; color: {text_color}; border: 1px solid #555;"
        )
        self.fancy_color_btn.setText(self._fancy_color)

    def _browse_watch_dir(self):
        """Browse for watch directory."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Watch Directory",
            self.watch_dir_edit.text() or str(Path.home())
        )
        if folder:
            self.watch_dir_edit.setText(folder)
            # Auto-set output if empty
            if not self.monitor_output_edit.text():
                self.monitor_output_edit.setText(str(Path(folder) / "fancy_bills"))
            # Auto-set archive if empty
            if not self.archive_dir_edit.text():
                self.archive_dir_edit.setText(str(Path(folder) / "archive"))

    def _browse_monitor_output(self):
        """Browse for monitor output directory."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Directory",
            self.monitor_output_edit.text() or str(Path.home())
        )
        if folder:
            self.monitor_output_edit.setText(folder)

    def _browse_archive_dir(self):
        """Browse for archive directory."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Archive Directory",
            self.archive_dir_edit.text() or str(Path.home())
        )
        if folder:
            self.archive_dir_edit.setText(folder)

    def _browse_review_dir(self):
        """Browse for review directory."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Review Directory",
            self.review_dir_edit.text() or str(Path.home())
        )
        if folder:
            self.review_dir_edit.setText(folder)
