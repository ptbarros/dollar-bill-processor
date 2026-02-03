"""
Recovery Dialog - Session recovery prompt on startup.

Shows when a recovery file is detected, allowing the user to:
- Restore the previous session
- Discard and start fresh
- View details about the recovery
"""

import sys
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QTextEdit, QDialogButtonBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class RecoveryDialog(QDialog):
    """Dialog shown on startup when a recovery file is detected."""

    # Result codes
    RESTORE = 1
    DISCARD = 2

    def __init__(self, recovery_info: dict, parent=None):
        """Initialize the recovery dialog.

        Args:
            recovery_info: Dictionary from SessionRecoveryManager.get_recovery_info()
            parent: Parent widget
        """
        super().__init__(parent)
        self.recovery_info = recovery_info
        self._result_action = None

        self.setWindowTitle("Session Recovery")
        self.setMinimumWidth(450)
        self.setModal(True)

        self._setup_ui()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Header with icon/message
        header_layout = QHBoxLayout()

        # Warning icon (using text for simplicity)
        icon_label = QLabel("\u26A0")  # Warning sign
        icon_font = QFont()
        icon_font.setPointSize(32)
        icon_label.setFont(icon_font)
        icon_label.setStyleSheet("color: #f0ad4e;")  # Warning yellow/orange
        header_layout.addWidget(icon_label)

        # Message
        msg_layout = QVBoxLayout()
        title_label = QLabel("Previous Session Found")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        msg_layout.addWidget(title_label)

        subtitle = QLabel(
            "A recovery file was found from a previous session.\n"
            "Would you like to restore your work?"
        )
        subtitle.setWordWrap(True)
        msg_layout.addWidget(subtitle)

        header_layout.addLayout(msg_layout, 1)
        layout.addLayout(header_layout)

        # Session info group
        info_group = QGroupBox("Session Details")
        info_layout = QFormLayout(info_group)

        # Format timestamp nicely
        timestamp_str = self.recovery_info.get("timestamp", "Unknown")
        try:
            dt = datetime.fromisoformat(timestamp_str)
            timestamp_display = dt.strftime("%Y-%m-%d %H:%M:%S")
            # Calculate age
            age = datetime.now() - dt
            if age.days > 0:
                age_str = f" ({age.days} day{'s' if age.days > 1 else ''} ago)"
            elif age.seconds > 3600:
                hours = age.seconds // 3600
                age_str = f" ({hours} hour{'s' if hours > 1 else ''} ago)"
            else:
                minutes = age.seconds // 60
                age_str = f" ({minutes} minute{'s' if minutes > 1 else ''} ago)"
            timestamp_display += age_str
        except (ValueError, TypeError):
            timestamp_display = timestamp_str

        time_label = QLabel(timestamp_display)
        info_layout.addRow("Last saved:", time_label)

        # Input directory
        input_dir = self.recovery_info.get("input_directory", "Unknown")
        if len(input_dir) > 50:
            input_dir = "..." + input_dir[-47:]
        input_label = QLabel(input_dir)
        input_label.setToolTip(self.recovery_info.get("input_directory", ""))
        info_layout.addRow("Input folder:", input_label)

        # Result count
        result_count = self.recovery_info.get("result_count", 0)
        total_processed = self.recovery_info.get("total_processed", result_count)
        results_label = QLabel(f"{result_count} bills")
        info_layout.addRow("Results:", results_label)

        # Processing status
        if self.recovery_info.get("processing_complete", False):
            status = "Complete"
            status_style = "color: #5cb85c;"  # Green
        else:
            status = "Interrupted"
            status_style = "color: #f0ad4e;"  # Yellow/orange
        status_label = QLabel(status)
        status_label.setStyleSheet(status_style)
        info_layout.addRow("Status:", status_label)

        layout.addWidget(info_group)

        # Warning about missing files (check input directory exists)
        input_path = Path(self.recovery_info.get("input_directory", ""))
        if input_path and not input_path.exists():
            warning_label = QLabel(
                "\u26A0 Warning: The original input folder no longer exists.\n"
                "File previews may not work if restored."
            )
            warning_label.setStyleSheet("color: #d9534f; font-style: italic;")
            warning_label.setWordWrap(True)
            layout.addWidget(warning_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        discard_btn = QPushButton("Discard && Start Fresh")
        discard_btn.clicked.connect(self._on_discard)
        button_layout.addWidget(discard_btn)

        restore_btn = QPushButton("Restore Session")
        restore_btn.setDefault(True)
        restore_btn.setStyleSheet(
            "QPushButton { background-color: #5cb85c; color: white; "
            "padding: 8px 16px; font-weight: bold; }"
            "QPushButton:hover { background-color: #4cae4c; }"
        )
        restore_btn.clicked.connect(self._on_restore)
        button_layout.addWidget(restore_btn)

        layout.addLayout(button_layout)

    def _on_restore(self):
        """Handle restore button click."""
        self._result_action = self.RESTORE
        self.accept()

    def _on_discard(self):
        """Handle discard button click."""
        self._result_action = self.DISCARD
        self.accept()

    def get_action(self) -> int:
        """Get the user's chosen action.

        Returns:
            RESTORE or DISCARD constant
        """
        return self._result_action


class RecoveryDetailsDialog(QDialog):
    """Dialog showing detailed recovery information."""

    def __init__(self, recovery_data: dict, parent=None):
        """Initialize the details dialog.

        Args:
            recovery_data: Full recovery data from load_recovery()
            parent: Parent widget
        """
        super().__init__(parent)
        self.recovery_data = recovery_data

        self.setWindowTitle("Recovery Details")
        self.setMinimumSize(600, 400)

        self._setup_ui()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)

        # Info text
        info = QTextEdit()
        info.setReadOnly(True)
        info.setFont(QFont("Monospace", 10))

        # Format the data
        lines = [
            f"Version: {self.recovery_data.get('version', 'Unknown')}",
            f"Timestamp: {self.recovery_data.get('timestamp', 'Unknown')}",
            f"Input Directory: {self.recovery_data.get('input_directory', 'Unknown')}",
            f"Processing Complete: {self.recovery_data.get('processing_complete', False)}",
            f"Total Processed: {self.recovery_data.get('total_processed', 0)}",
            f"Results Count: {len(self.recovery_data.get('results', []))}",
            "",
            "Results Preview (first 10):",
            "-" * 50,
        ]

        results = self.recovery_data.get("results", [])[:10]
        for i, r in enumerate(results, 1):
            serial = r.get("serial", "N/A")
            fancy = r.get("fancy_types", "")
            is_fancy = r.get("is_fancy", False)
            lines.append(f"{i}. {serial} {'[FANCY: ' + fancy + ']' if is_fancy else ''}")

        if len(self.recovery_data.get("results", [])) > 10:
            lines.append(f"... and {len(self.recovery_data.get('results', [])) - 10} more")

        info.setPlainText("\n".join(lines))
        layout.addWidget(info)

        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
