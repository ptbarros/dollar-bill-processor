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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Input folder selection
        input_group = QFrame()
        input_layout = QHBoxLayout(input_group)
        input_layout.setContentsMargins(0, 0, 0, 0)

        input_label = QLabel("Input:")
        input_label.setFixedWidth(50)
        input_layout.addWidget(input_label)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Select folder with scanned bills...")
        self.input_edit.setMinimumWidth(200)
        input_layout.addWidget(self.input_edit)

        self.browse_input_btn = QPushButton("Browse...")
        self.browse_input_btn.clicked.connect(self._browse_input)
        input_layout.addWidget(self.browse_input_btn)

        layout.addWidget(input_group, 1)

        # Output folder selection
        output_group = QFrame()
        output_layout = QHBoxLayout(output_group)
        output_layout.setContentsMargins(0, 0, 0, 0)

        output_label = QLabel("Output:")
        output_label.setFixedWidth(50)
        output_layout.addWidget(output_label)

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Output folder for fancy bills...")
        self.output_edit.setMinimumWidth(150)
        output_layout.addWidget(self.output_edit)

        self.browse_output_btn = QPushButton("Browse...")
        self.browse_output_btn.clicked.connect(self._browse_output)
        output_layout.addWidget(self.browse_output_btn)

        layout.addWidget(output_group, 1)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # Process/Stop buttons
        self.process_btn = QPushButton("Process")
        self.process_btn.setMinimumWidth(80)
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

    def set_processing(self, is_processing: bool):
        """Update UI for processing state."""
        self.process_btn.setEnabled(not is_processing)
        self.stop_btn.setEnabled(is_processing)
        self.browse_input_btn.setEnabled(not is_processing)
        self.browse_output_btn.setEnabled(not is_processing)
        self.input_edit.setEnabled(not is_processing)
        self.output_edit.setEnabled(not is_processing)

        if not is_processing:
            self.progress_bar.setValue(0)

    def update_progress(self, current: int, total: int):
        """Update progress bar."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
