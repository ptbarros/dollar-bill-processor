"""
Processing Panel - Top toolbar for processing controls.
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLineEdit,
    QProgressBar, QLabel, QFileDialog, QFrame, QComboBox, QMessageBox,
    QCheckBox, QButtonGroup
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
    watch_toggled = Signal(bool)  # Monitor: on/off (watches the configured Data Folder)
    live_toggled = Signal(bool)   # EXPERIMENTAL: process scans live while scanning
    mode_changed = Signal(str)    # "manual" or "scan" — toolbar reconfigured
    open_folders_settings = Signal()  # gear next to the Watching indicator

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Mode toggle (segmented): Manual processing vs Scan straps. Reconfigures
        # the toolbar below; persisted in settings.ui.panel_mode. Only the checked
        # state is styled so the idle button keeps the themed look (a partial base
        # rule renders flat on Windows).
        _seg_style = ("QPushButton:checked { background-color: #2a82da; "
                      "color: white; font-weight: bold; }")
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.manual_mode_btn = QPushButton("Manual")
        self.manual_mode_btn.setToolTip("Manual processing: pick an input folder and Process it")
        self.scan_mode_btn = QPushButton("Scan")
        self.scan_mode_btn.setToolTip("Scan straps: watch the Scanner Output Folder and file straps")
        for _b in (self.manual_mode_btn, self.scan_mode_btn):
            _b.setCheckable(True)
            _b.setStyleSheet(_seg_style)
            self.mode_group.addButton(_b)
        self.manual_mode_btn.clicked.connect(lambda: self._on_mode_clicked("manual"))
        self.scan_mode_btn.clicked.connect(lambda: self._on_mode_clicked("scan"))
        _seg = QHBoxLayout()
        _seg.setSpacing(0)
        _seg.addWidget(self.manual_mode_btn)
        _seg.addWidget(self.scan_mode_btn)
        layout.addLayout(_seg)

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep1)

        # Scan-mode "Watching: <folder> · next strap: N" indicator (read-only; the
        # folder + naming live in Settings → Folders). Hidden in Manual mode.
        self.watch_info_group = QFrame()
        _wi = QHBoxLayout(self.watch_info_group)
        _wi.setContentsMargins(0, 0, 0, 0)
        self.watching_label = QLabel("Watching: …")
        self.watching_label.setStyleSheet("color: #2a82da;")
        self._watching_full = ""
        _wi.addWidget(self.watching_label)
        self.watch_settings_btn = QPushButton("⚙")
        self.watch_settings_btn.setMaximumWidth(28)
        self.watch_settings_btn.setToolTip(
            "Change the Scanner Output Folder and strap naming in Settings → Folders")
        self.watch_settings_btn.clicked.connect(self.open_folders_settings.emit)
        _wi.addWidget(self.watch_settings_btn)
        # No stretch here: it would expand and push the Scan buttons off-screen.
        layout.addWidget(self.watch_info_group)

        # Input folder selection (manual mode)
        self.input_group = QFrame()
        input_layout = QHBoxLayout(self.input_group)
        input_layout.setContentsMargins(0, 0, 0, 0)

        self.input_label = QLabel("Input:")
        input_layout.addWidget(self.input_label)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Select folder with scanned bills...")
        self.input_edit.setMinimumWidth(110)
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
        self.output_edit.setMinimumWidth(110)
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

        # Watch Folder (Monitor mode): when on, scans dropped into the selected
        # folder are auto-filed into a Straps batch and processed as they settle.
        self.watch_btn = QPushButton("Start Scanning")
        self.watch_btn.setCheckable(True)
        # Size to the WIDEST label ("Stop && File Batch", shown while scanning and
        # drawn bold) so neither state clips. '&&' renders as a single '&'; add a
        # cushion for the bold weight + button padding.
        _fm = self.watch_btn.fontMetrics()
        _watch_w = max(_fm.horizontalAdvance("Start Scanning"),
                       _fm.horizontalAdvance("Stop & File Batch"))
        self.watch_btn.setMinimumWidth(_watch_w + 56)
        self.watch_btn.setToolTip(
            "Start collecting scans as they come off the scanner (feed your whole "
            "strap in as many passes as you like). Click Stop when the strap is "
            "done to file everything into one batch under Straps and process it.")
        # Only style the "on" (checked) state blue. The idle state deliberately has
        # NO base QPushButton rule so it inherits the app's themed button look
        # (background + border) -- a partial base rule strips the native chrome and
        # makes the button render as flat text on Windows.
        self.watch_btn.setStyleSheet("""
            QPushButton:checked {
                background-color: #2a82da; color: white; font-weight: bold;
                border: 1px solid #2a82da;
            }
            QPushButton:checked:hover { background-color: #2372c4; }
        """)
        self.watch_btn.toggled.connect(self._on_watch_toggled)
        layout.addWidget(self.watch_btn)

        # EXPERIMENTAL: process scans live (in small chunks) while scanning,
        # instead of waiting for Stop. Can't be changed mid-scan.
        self.live_check = QCheckBox("Process live")
        self.live_check.setToolTip(
            "EXPERIMENTAL: process scans in small batches as they come in, "
            "instead of all at once when you click Stop. Set this before you "
            "click Start Scanning.")
        try:
            self.live_check.setChecked(bool(get_settings().processing.live_processing))
        except Exception:
            pass
        self.live_check.toggled.connect(self._on_live_toggled)
        layout.addWidget(self.live_check)

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

        # Apply the saved mode (show/hide the right widgets) without persisting.
        try:
            self.set_mode(get_settings().ui.panel_mode, persist=False)
        except Exception:
            self.set_mode("manual", persist=False)

    # ---- Mode (Manual processing vs Scan straps) ----------------------------

    def set_mode(self, mode: str, persist: bool = True):
        """Reconfigure the toolbar for `mode` ("manual" or "scan"). Persists to
        settings unless persist=False (e.g. applying the saved value at startup)."""
        mode = "scan" if mode == "scan" else "manual"
        self._mode = mode
        self.manual_mode_btn.setChecked(mode == "manual")
        self.scan_mode_btn.setChecked(mode == "scan")
        manual = (mode == "manual")
        # Manual-only widgets
        self.input_group.setVisible(manual)
        self.output_group.setVisible(manual)
        self.process_btn.setVisible(manual)
        self.archive_btn.setVisible(manual)
        # Scan-only widgets
        self.watch_btn.setVisible(not manual)
        self.live_check.setVisible(not manual)
        self.watch_info_group.setVisible(not manual)
        if persist:
            try:
                get_settings().ui.panel_mode = mode
                get_settings().save()
            except Exception:
                pass
        self.mode_changed.emit(mode)

    def _on_mode_clicked(self, mode: str):
        """A mode button was clicked. Block switching mid-run (revert the toggle)."""
        if getattr(self, "_watching", False) or getattr(self, "_is_processing", False):
            self.set_mode(getattr(self, "_mode", "manual"), persist=False)
            return
        self.set_mode(mode)

    def set_watching_info(self, text: str):
        """Set the Scan-mode 'Watching: …' indicator (MainWindow builds it). Elide
        the middle so a long path can't blow up the toolbar width; full text goes
        in the tooltip."""
        self._watching_full = text
        self.watching_label.setToolTip(text)
        fm = self.watching_label.fontMetrics()
        self.watching_label.setText(fm.elidedText(text, Qt.ElideMiddle, 460))

    def current_mode(self) -> str:
        return getattr(self, "_mode", "manual")

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
            if not current_output or (old_input and current_output == self._default_output_for(old_input)):
                self.output_edit.setText(self._default_output_for(folder))

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

    def _refresh_process_enabled(self):
        """Process (a manual run) is available only when we're neither scanning
        nor already processing -- both would conflict with a manual run."""
        watching = getattr(self, "_watching", False)
        processing = getattr(self, "_is_processing", False)
        self.process_btn.setEnabled(not watching and not processing)
        # Can't switch modes mid-run.
        busy = watching or processing
        if hasattr(self, "manual_mode_btn"):
            self.manual_mode_btn.setEnabled(not busy)
            self.scan_mode_btn.setEnabled(not busy)

    def _on_live_toggled(self, checked: bool):
        """Persist the experimental live-processing toggle and notify listeners."""
        try:
            get_settings().processing.live_processing = checked
            get_settings().save()
        except Exception:
            pass
        self.live_toggled.emit(checked)

    def _on_watch_toggled(self, checked: bool):
        """Start/stop watching the Data Folder for new scans. (The button text
        reflects what a click will do; MainWindow owns the watch folder.)"""
        self.watch_btn.setText("Stop && File Batch" if checked else "Start Scanning")
        self._watching = checked
        # Live mode can't be flipped mid-scan (it changes how a run is wired).
        self.live_check.setEnabled(not checked)
        self._refresh_process_enabled()
        self.watch_toggled.emit(checked)

    def set_watching(self, on: bool):
        """Reflect watch state on the button without re-emitting (used when
        MainWindow declines to start, e.g. a missing folder)."""
        self.watch_btn.blockSignals(True)
        self.watch_btn.setChecked(on)
        self.watch_btn.setText("Stop && File Batch" if on else "Start Scanning")
        self.watch_btn.blockSignals(False)
        self._watching = on
        self.live_check.setEnabled(not on)
        self._refresh_process_enabled()

    def _on_process(self):
        """Handle process button click."""
        input_dir = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip()

        if not input_dir:
            return

        if not output_dir:
            output_dir = self._default_output_for(input_dir)
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
        """Public entry point for the Edit -> Organize Folder menu action.

        Unlike the raw handler, this explains what Organize does (restoring the
        info the old toolbar button showed) and confirms before running."""
        input_dir = self.input_edit.text().strip()
        if not input_dir:
            QMessageBox.information(
                self, "Organize Folder",
                "Choose an input folder first, then run Organize Folder to "
                "pre-process it.")
            return

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Organize Folder")
        box.setText("Pre-process this folder to speed up re-processing?")
        box.setInformativeText(
            "Organize scans the input folder and, in place, will:\n"
            "  • classify each image as front or back\n"
            "  • fix upside-down images and correct skew\n"
            "  • rename files to Dollar_NNN.jpg (odd = front, even = back)\n\n"
            "Afterwards, processing runs faster because the verify and "
            "alignment steps can be skipped.\n\n"
            f"Folder:\n{input_dir}")
        box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        box.button(QMessageBox.Ok).setText("Organize")
        box.setDefaultButton(QMessageBox.Cancel)
        if box.exec() != QMessageBox.Ok:
            return
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
            self.output_edit.setText(self._default_output_for(path))

    def set_output_dir(self, path: str):
        """Set the output directory."""
        self.output_edit.setText(path)

    def _output_subfolder(self) -> str:
        """The configurable output subfolder name (Settings -> Folders),
        defaulting to 'fancy_bills'."""
        name = (get_settings().processing.output_subfolder or "").strip()
        return name or "fancy_bills"

    def _default_output_for(self, input_dir: str) -> str:
        """The auto-generated output path for an input folder."""
        return str(Path(input_dir) / self._output_subfolder())

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
        self._is_processing = is_processing
        self._refresh_process_enabled()
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
