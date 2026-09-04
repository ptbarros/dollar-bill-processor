"""
Processing Panel - Top toolbar for processing controls.
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLineEdit,
    QProgressBar, QLabel, QFileDialog, QFrame, QComboBox
)
from PySide6.QtCore import Qt, Signal, Slot

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from settings_manager import get_settings


class ProcessingPanel(QWidget):
    """Panel containing processing controls and progress."""

    # Signals
    process_requested = Signal(str, str)  # input_dir, output_dir
    organize_requested = Signal(str)  # input_dir - organize folder before processing
    profile_changed = Signal(str)  # active crop profile picked from the toolbar
    stop_requested = Signal()
    archive_requested = Signal()  # Archive the current batch

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep1)

        # Input folder selection (manual mode)
        self.input_group = QFrame()
        input_layout = QHBoxLayout(self.input_group)
        input_layout.setContentsMargins(0, 0, 0, 0)

        self.input_label = QLabel("Input:")
        input_layout.addWidget(self.input_label)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Select folder with scanned bills...")
        self.input_edit.setMinimumWidth(200)
        input_layout.addWidget(self.input_edit)

        self.browse_input_btn = QPushButton("Browse...")
        self.browse_input_btn.clicked.connect(self._browse_input)
        input_layout.addWidget(self.browse_input_btn)

        layout.addWidget(self.input_group, 1)

        # Output folder selection (manual mode)
        self.output_group = QFrame()
        output_layout = QHBoxLayout(self.output_group)
        output_layout.setContentsMargins(0, 0, 0, 0)

        self.output_label = QLabel("Output:")
        output_layout.addWidget(self.output_label)

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Output folder for fancy bills...")
        self.output_edit.setMinimumWidth(150)
        output_layout.addWidget(self.output_edit)

        self.browse_output_btn = QPushButton("Browse...")
        self.browse_output_btn.clicked.connect(self._browse_output)
        output_layout.addWidget(self.browse_output_btn)

        layout.addWidget(self.output_group, 1)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # Process/Stop buttons
        self.process_btn = QPushButton("Process")
        self.process_btn.setMinimumWidth(100)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.process_btn.clicked.connect(self._on_process)
        layout.addWidget(self.process_btn)

        # Active crop-profile picker (replaces the old Organize button; Organize
        # moved to Edit -> Organize Folder). Switching here changes the profile
        # and denomination used for the next Process run.
        self.profile_label = QLabel("Profile:")
        layout.addWidget(self.profile_label)
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(120)
        self.profile_combo.setToolTip(
            "Active crop profile (and its denomination) used for processing.\n"
            "Create and edit profiles in the Crop Manager.")
        self.profile_combo.currentIndexChanged.connect(self._on_profile_combo_changed)
        layout.addWidget(self.profile_combo)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumWidth(60)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.stop_btn.clicked.connect(self._on_stop)
        layout.addWidget(self.stop_btn)

        # Archive button - for manual archiving after processing
        self.archive_btn = QPushButton("Archive")
        self.archive_btn.setMinimumWidth(60)
        self.archive_btn.setEnabled(False)
        self.archive_btn.setToolTip("Move processed files to archive folder")
        self.archive_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.archive_btn.clicked.connect(self.archive_requested.emit)
        layout.addWidget(self.archive_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumWidth(150)
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v/%m")
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

    def _browse_input(self):
        """Browse for input folder."""
        settings = get_settings()
        # Priority: current field > default_working_dir > home
        start_dir = (self.input_edit.text() or
                     settings.ui.default_working_dir or
                     str(Path.home()))
        folder = QFileDialog.getExistingDirectory(
            self, "Select Input Folder", start_dir
        )
        if folder:
            old_input = self.input_edit.text()
            self.input_edit.setText(folder)

            # Auto-update output if it's empty or still matches the old auto-generated path
            current_output = self.output_edit.text()
            if not current_output or (old_input and current_output == str(Path(old_input) / "fancy_bills")):
                self.output_edit.setText(str(Path(folder) / "fancy_bills"))

    def _browse_output(self):
        """Browse for output folder."""
        settings = get_settings()
        # Priority: current field > default_working_dir > home
        start_dir = (self.output_edit.text() or
                     settings.ui.default_working_dir or
                     str(Path.home()))
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", start_dir
        )
        if folder:
            self.output_edit.setText(folder)

    def _on_process(self):
        """Handle process button click."""
        input_dir = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip()

        if not input_dir:
            return

        if not output_dir:
            output_dir = str(Path(input_dir) / "fancy_bills")
            self.output_edit.setText(output_dir)

        self.process_requested.emit(input_dir, output_dir)

    def _on_organize(self):
        """Handle organize button click."""
        input_dir = self.input_edit.text().strip()

        if not input_dir:
            return

        print(f"[ProcessingPanel] Emitting organize_requested({input_dir})")
        self.organize_requested.emit(input_dir)

    def trigger_organize(self):
        """Public entry point for the Edit -> Organize Folder menu action."""
        self._on_organize()

    def set_profiles(self, names, active_name):
        """Populate the toolbar profile picker and select the active profile."""
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        for n in names:
            self.profile_combo.addItem(n)
        if active_name and active_name in list(names):
            self.profile_combo.setCurrentText(active_name)
        self.profile_combo.blockSignals(False)

    def _on_profile_combo_changed(self, _idx):
        name = self.profile_combo.currentText()
        if name:
            self.profile_changed.emit(name)

    def _on_stop(self):
        """Handle stop button click."""
        self.stop_requested.emit()

    def set_input_dir(self, path: str):
        """Set the input directory."""
        self.input_edit.setText(path)
        if not self.output_edit.text():
            self.output_edit.setText(str(Path(path) / "fancy_bills"))

    def set_output_dir(self, path: str):
        """Set the output directory."""
        self.output_edit.setText(path)

    def set_denomination(self, denom):
        """Show the active profile's denomination on the Process button so it's
        visible right before processing (e.g. 'Process $5')."""
        try:
            d = int(denom)
        except (TypeError, ValueError):
            d = 1
        self.process_btn.setText(f"Process ${d}")

    def set_processing(self, is_processing: bool):
        """Update UI for processing state."""
        self.process_btn.setEnabled(not is_processing)
        self.profile_combo.setEnabled(not is_processing)
        self.stop_btn.setEnabled(is_processing)

        self.browse_input_btn.setEnabled(not is_processing)
        self.browse_output_btn.setEnabled(not is_processing)
        self.input_edit.setEnabled(not is_processing)
        self.output_edit.setEnabled(not is_processing)

        if not is_processing:
            self.progress_bar.setValue(0)

        # Disable archive button during processing
        if is_processing:
            self.archive_btn.setEnabled(False)

    def set_archive_available(self, available: bool, auto_archive_enabled: bool):
        """Update archive button state after processing completes.

        Args:
            available: Whether there are results to archive
            auto_archive_enabled: Whether auto-archive is enabled in settings
        """
        print(f"[ProcessingPanel] set_archive_available(available={available}, auto_archive_enabled={auto_archive_enabled})")
        if auto_archive_enabled:
            # Auto-archive is on, so hide/disable the manual button
            self.archive_btn.setEnabled(False)
            self.archive_btn.setToolTip("Auto-archive is enabled in settings")
        else:
            # Manual archive available
            self.archive_btn.setEnabled(available)
            self.archive_btn.setToolTip("Move processed files to archive folder")

    def reset_archive_button(self):
        """Reset archive button to disabled state (e.g., after archiving)."""
        self.archive_btn.setEnabled(False)



    def update_progress(self, current: int, total: int):
        """Update progress bar."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
