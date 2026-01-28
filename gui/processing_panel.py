"""
Processing Panel - Top toolbar for processing controls.
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLineEdit,
    QProgressBar, QLabel, QFileDialog, QFrame
)
from PySide6.QtCore import Qt, Signal, Slot


class ProcessingPanel(QWidget):
    """Panel containing processing controls and progress."""

    # Signals
    process_requested = Signal(str, str)  # input_dir, output_dir
    stop_requested = Signal()
    monitor_requested = Signal()  # Start monitoring
    monitor_stop_requested = Signal()  # Stop monitoring

    def __init__(self, parent=None):
        super().__init__(parent)
        self._monitor_mode = False
        self._is_monitoring = False
        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Monitor mode checkbox
        from PySide6.QtWidgets import QCheckBox
        self.monitor_check = QCheckBox("Monitor")
        self.monitor_check.setToolTip("Enable monitor mode to watch a directory for incoming files")
        self.monitor_check.toggled.connect(self._on_monitor_toggled)
        layout.addWidget(self.monitor_check)

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

        # Monitor mode display (hidden by default)
        self.monitor_group = QFrame()
        monitor_layout = QHBoxLayout(self.monitor_group)
        monitor_layout.setContentsMargins(0, 0, 0, 0)

        self.monitor_status_label = QLabel()
        self.monitor_status_label.setStyleSheet("color: #666;")
        monitor_layout.addWidget(self.monitor_status_label)

        # Monitor indicator (blinking dot when active)
        self.monitor_indicator = QLabel()
        self.monitor_indicator.setFixedSize(12, 12)
        self.monitor_indicator.setStyleSheet(
            "background-color: #888; border-radius: 6px;"
        )
        monitor_layout.addWidget(self.monitor_indicator)

        layout.addWidget(self.monitor_group, 1)
        self.monitor_group.hide()

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
        folder = QFileDialog.getExistingDirectory(
            self, "Select Input Folder",
            self.input_edit.text() or str(Path.home())
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
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder",
            self.output_edit.text() or str(Path.home())
        )
        if folder:
            self.output_edit.setText(folder)

    def _on_monitor_toggled(self, checked: bool):
        """Handle monitor mode checkbox toggle."""
        self._monitor_mode = checked

        if checked:
            # Switch to monitor mode display
            self.input_group.hide()
            self.output_group.hide()
            self.monitor_group.show()
            self.process_btn.setText("Start Monitoring")
            self._update_monitor_display()
        else:
            # Switch to manual mode display
            self.monitor_group.hide()
            self.input_group.show()
            self.output_group.show()
            self.process_btn.setText("Process")

    def _update_monitor_display(self, watch_dir: str = "", output_dir: str = ""):
        """Update the monitor mode display labels."""
        if watch_dir:
            self.monitor_status_label.setText(f"Watch: {watch_dir}")
        else:
            self.monitor_status_label.setText("Configure directories in Settings > Monitor")

    def set_monitor_dirs(self, watch_dir: str, output_dir: str):
        """Set monitor directories for display."""
        if watch_dir:
            self.monitor_status_label.setText(f"Watching: {watch_dir}")
        else:
            self.monitor_status_label.setText("Configure directories in Settings > Monitor")

    def _on_process(self):
        """Handle process button click."""
        if self._monitor_mode:
            if self._is_monitoring:
                # Already monitoring - this shouldn't happen as button should be disabled
                return
            self.monitor_requested.emit()
        else:
            input_dir = self.input_edit.text().strip()
            output_dir = self.output_edit.text().strip()

            if not input_dir:
                return

            if not output_dir:
                output_dir = str(Path(input_dir) / "fancy_bills")
                self.output_edit.setText(output_dir)

            self.process_requested.emit(input_dir, output_dir)

    def _on_stop(self):
        """Handle stop button click."""
        if self._monitor_mode:
            self.monitor_stop_requested.emit()
        else:
            self.stop_requested.emit()

    def set_input_dir(self, path: str):
        """Set the input directory."""
        self.input_edit.setText(path)
        if not self.output_edit.text():
            self.output_edit.setText(str(Path(path) / "fancy_bills"))

    def set_output_dir(self, path: str):
        """Set the output directory."""
        self.output_edit.setText(path)

    def set_processing(self, is_processing: bool):
        """Update UI for processing state."""
        self.process_btn.setEnabled(not is_processing)
        self.stop_btn.setEnabled(is_processing)
        self.monitor_check.setEnabled(not is_processing)

        if not self._monitor_mode:
            self.browse_input_btn.setEnabled(not is_processing)
            self.browse_output_btn.setEnabled(not is_processing)
            self.input_edit.setEnabled(not is_processing)
            self.output_edit.setEnabled(not is_processing)

        if not is_processing:
            self.progress_bar.setValue(0)

    def set_monitoring(self, is_monitoring: bool):
        """Update UI for monitoring state."""
        self._is_monitoring = is_monitoring
        self.process_btn.setEnabled(not is_monitoring)
        self.stop_btn.setEnabled(is_monitoring)
        self.monitor_check.setEnabled(not is_monitoring)

        # Update indicator color
        if is_monitoring:
            self.monitor_indicator.setStyleSheet(
                "background-color: #4CAF50; border-radius: 6px;"
            )
            self.process_btn.setText("Monitoring...")
        else:
            self.monitor_indicator.setStyleSheet(
                "background-color: #888; border-radius: 6px;"
            )
            self.process_btn.setText("Start Monitoring")

        if not is_monitoring:
            self.progress_bar.setValue(0)

    def is_monitor_mode(self) -> bool:
        """Return whether monitor mode is enabled."""
        return self._monitor_mode

    def update_progress(self, current: int, total: int):
        """Update progress bar."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
