"""
Main Window - Dollar Bill Processor GUI
The central window containing all GUI components.
"""

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QStatusBar, QMenuBar, QMenu, QFileDialog, QMessageBox,
    QProgressBar, QLabel, QPushButton
)
from PySide6.QtCore import Qt, Signal, Slot, QThread
from PySide6.QtGui import QAction, QKeySequence

# Import our components
from .processing_panel import ProcessingPanel
from .results_list import ResultsList
from .preview_panel import PreviewPanel

# Import backend
sys.path.insert(0, str(Path(__file__).parent.parent))
from settings_manager import SettingsManager, get_settings
from correction_manager import CorrectionManager


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()

        # Load settings
        self.settings = get_settings()
        self.correction_manager = CorrectionManager()

        # Processing state
        self.processor = None
        self.current_results = []
        self.is_processing = False

        # Setup UI
        self._setup_ui()
        self._setup_menus()
        self._setup_statusbar()
        self._restore_geometry()

        self.setWindowTitle("Dollar Bill Processor")

    def _setup_ui(self):
        """Setup the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Top toolbar area
        self.processing_panel = ProcessingPanel()
        self.processing_panel.process_requested.connect(self._on_process_requested)
        self.processing_panel.stop_requested.connect(self._on_stop_requested)
        main_layout.addWidget(self.processing_panel)

        # Main content area (splitter)
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Results list
        self.results_list = ResultsList()
        self.results_list.item_selected.connect(self._on_result_selected)
        self.results_list.correction_requested.connect(self._on_correction_requested)
        splitter.addWidget(self.results_list)

        # Right panel - Preview
        self.preview_panel = PreviewPanel()
        self.preview_panel.correction_submitted.connect(self._on_correction_submitted)
        splitter.addWidget(self.preview_panel)

        # Set splitter sizes (40% list, 60% preview)
        splitter.setSizes([400, 600])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter, 1)

    def _setup_menus(self):
        """Setup the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Folder...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._on_open_folder)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        export_menu = file_menu.addMenu("&Export Results")

        export_csv = QAction("Export as CSV...", self)
        export_csv.triggered.connect(lambda: self._export_results("csv"))
        export_menu.addAction(export_csv)

        export_excel = QAction("Export as Excel...", self)
        export_excel.triggered.connect(lambda: self._export_results("excel"))
        export_menu.addAction(export_excel)

        export_html = QAction("Export as HTML Report...", self)
        export_html.triggered.connect(lambda: self._export_results("html"))
        export_menu.addAction(export_html)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        settings_action = QAction("&Settings...", self)
        settings_action.setShortcut(QKeySequence("Ctrl+,"))
        settings_action.triggered.connect(self._on_settings)
        edit_menu.addAction(settings_action)

        edit_menu.addSeparator()

        patterns_action = QAction("&Pattern Manager...", self)
        patterns_action.triggered.connect(self._on_pattern_manager)
        edit_menu.addAction(patterns_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        review_only = QAction("Show &Review Items Only", self, checkable=True)
        review_only.triggered.connect(self._toggle_review_filter)
        view_menu.addAction(review_only)

        fancy_only = QAction("Show &Fancy Bills Only", self, checkable=True)
        fancy_only.triggered.connect(self._toggle_fancy_filter)
        view_menu.addAction(fancy_only)

        view_menu.addSeparator()

        refresh_action = QAction("&Refresh", self)
        refresh_action.setShortcut(QKeySequence.Refresh)
        refresh_action.triggered.connect(self._refresh_view)
        view_menu.addAction(refresh_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About...", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _setup_statusbar(self):
        """Setup the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        # Status label
        self.status_label = QLabel("Ready")
        self.statusbar.addWidget(self.status_label, 1)

        # Progress info
        self.progress_label = QLabel("")
        self.statusbar.addPermanentWidget(self.progress_label)

    def _restore_geometry(self):
        """Restore window geometry from settings."""
        x, y, w, h = self.settings.get_window_geometry()
        self.setGeometry(x, y, w, h)

    def _save_geometry(self):
        """Save window geometry to settings."""
        geo = self.geometry()
        self.settings.update_window_geometry(geo.x(), geo.y(), geo.width(), geo.height())
        self.settings.save()

    def closeEvent(self, event):
        """Handle window close."""
        self._save_geometry()
        if self.is_processing:
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "Processing is in progress. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            self._on_stop_requested()
        event.accept()

    # Slots
    @Slot(str, str)
    def _on_process_requested(self, input_dir: str, output_dir: str):
        """Handle process request from processing panel."""
        self.is_processing = True
        self.status_label.setText(f"Processing: {input_dir}")
        self.processing_panel.set_processing(True)

        # Clear previous results when starting a new batch
        self.current_results = []
        self.results_list.clear()

        # Save directories to settings
        self.settings.ui.last_input_dir = input_dir
        self.settings.ui.last_output_dir = output_dir
        self.settings.save()

        # Start processing in background thread
        self._start_processing(input_dir, output_dir)

    @Slot()
    def _on_stop_requested(self):
        """Handle stop request."""
        self.is_processing = False
        self.status_label.setText("Processing stopped")
        self.processing_panel.set_processing(False)
        # TODO: Signal the processing thread to stop

    @Slot(dict)
    def _on_result_selected(self, result: dict):
        """Handle selection of a result item."""
        self.preview_panel.show_bill(result)

    @Slot(dict)
    def _on_correction_requested(self, result: dict):
        """Handle correction request for a result."""
        self.preview_panel.show_bill(result)
        self.preview_panel.start_correction()

    @Slot(str, str, str)
    def _on_correction_submitted(self, filename: str, original: str, corrected: str):
        """Handle a submitted correction."""
        self.correction_manager.add_correction(filename, original, corrected)
        self.correction_manager.save()

        # Update the result in our list
        for result in self.current_results:
            if result.get('front_file') == filename:
                result['serial'] = corrected
                result['corrected'] = True
                break

        # Refresh the display, preserving selection
        self.results_list.refresh()
        self.results_list.select_by_filename(filename)
        self.status_label.setText(f"Correction saved: {filename}")

    def _on_open_folder(self):
        """Handle open folder action."""
        start_dir = self.settings.ui.last_input_dir or str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            self, "Select Scan Folder", start_dir
        )
        if folder:
            self.processing_panel.set_input_dir(folder)

    def _on_settings(self):
        """Open settings dialog."""
        from .settings_dialog import SettingsDialog
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            self.settings.save()
            self._apply_settings()

    def _on_pattern_manager(self):
        """Open pattern manager dialog."""
        from .pattern_dialog import PatternDialog
        dialog = PatternDialog(self)
        dialog.exec()

    def _toggle_review_filter(self, checked: bool):
        """Toggle showing only review items."""
        self.results_list.set_filter('needs_review', checked)

    def _toggle_fancy_filter(self, checked: bool):
        """Toggle showing only fancy bills."""
        self.results_list.set_filter('is_fancy', checked)

    def _refresh_view(self):
        """Refresh the current view."""
        self.results_list.refresh()

    def _on_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About Dollar Bill Processor",
            "Dollar Bill Processor v1.0\n\n"
            "Automated detection of fancy serial numbers\n"
            "on US currency bills.\n\n"
            "Features:\n"
            "- YOLO-based serial number detection\n"
            "- 50+ pattern recognition rules\n"
            "- Manual correction workflow\n\n"
            "Built with PySide6 and OpenCV"
        )

    def _export_results(self, format_type: str):
        """Export results to file."""
        if not self.current_results:
            QMessageBox.warning(self, "No Results", "No results to export.")
            return

        if format_type == "csv":
            path, _ = QFileDialog.getSaveFileName(
                self, "Export CSV", "", "CSV Files (*.csv)"
            )
            if path:
                self._export_csv(path)
        elif format_type == "excel":
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Excel", "", "Excel Files (*.xlsx)"
            )
            if path:
                self._export_excel(path)
        elif format_type == "html":
            path, _ = QFileDialog.getSaveFileName(
                self, "Export HTML", "", "HTML Files (*.html)"
            )
            if path:
                self._export_html(path)

    def _auto_export(self, summary: dict):
        """Auto-export CSV and/or summary if enabled in settings."""
        from datetime import datetime
        input_dir = Path(self.settings.ui.last_input_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        exported = []

        # Auto-export CSV
        if self.settings.export.auto_export_csv:
            csv_path = input_dir / f"results_{timestamp}.csv"
            self._export_csv(str(csv_path), quiet=True)
            exported.append(f"CSV: {csv_path.name}")

        # Auto-export summary
        if self.settings.export.auto_export_summary:
            summary_path = input_dir / f"summary_{timestamp}.txt"
            self._export_summary(str(summary_path), summary)
            exported.append(f"Summary: {summary_path.name}")

        if exported:
            self.status_label.setText(
                f"Complete: {summary.get('total', 0)} bills | Auto-exported: {', '.join(exported)}"
            )

    def _export_summary(self, path: str, summary: dict):
        """Export processing summary to text file."""
        from datetime import datetime
        fancy_bills = [r for r in self.current_results if r.get('is_fancy')]

        with open(path, 'w') as f:
            f.write("Dollar Bill Processing Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input: {self.settings.ui.last_input_dir}\n\n")

            f.write(f"Total bills processed: {summary.get('total', 0)}\n")
            f.write(f"Fancy bills found: {summary.get('fancy_count', 0)}\n")
            f.write(f"Bills needing review: {summary.get('review_count', 0)}\n\n")

            if fancy_bills:
                f.write("Fancy Bills:\n")
                f.write("-" * 40 + "\n")
                for bill in fancy_bills:
                    f.write(f"  {bill.get('serial', 'N/A')}: {bill.get('fancy_types', '')}\n")

    def _export_csv(self, path: str, quiet: bool = False):
        """Export results to CSV."""
        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'position', 'front_file', 'back_file', 'serial', 'fancy_types',
                'confidence', 'is_fancy', 'needs_review', 'serial_region_path', 'error'
            ])
            writer.writeheader()
            writer.writerows(self.current_results)
        if not quiet:
            self.status_label.setText(f"Exported to {path}")

    def _export_excel(self, path: str):
        """Export results to Excel."""
        try:
            import pandas as pd
            df = pd.DataFrame(self.current_results)
            df.to_excel(path, index=False)
            self.status_label.setText(f"Exported to {path}")
        except ImportError:
            QMessageBox.warning(
                self, "Missing Dependency",
                "pandas and openpyxl are required for Excel export.\n"
                "Install with: pip install pandas openpyxl"
            )

    def _export_html(self, path: str):
        """Export results to HTML report."""
        html = self._generate_html_report()
        with open(path, 'w') as f:
            f.write(html)
        self.status_label.setText(f"Exported to {path}")

    def _generate_html_report(self) -> str:
        """Generate HTML report content."""
        fancy_bills = [r for r in self.current_results if r.get('is_fancy')]
        review_bills = [r for r in self.current_results if r.get('needs_review')]

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Dollar Bill Processing Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2e7d32; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .fancy {{ background-color: #c8e6c9; }}
        .review {{ background-color: #fff3e0; }}
        .summary {{ background-color: #e3f2fd; padding: 15px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>Dollar Bill Processing Report</h1>

    <div class="summary">
        <strong>Summary:</strong><br>
        Total Bills: {len(self.current_results)}<br>
        Fancy Bills: {len(fancy_bills)}<br>
        Needs Review: {len(review_bills)}
    </div>

    <h2>Fancy Bills</h2>
    <table>
        <tr><th>Position</th><th>Serial</th><th>Patterns</th><th>Confidence</th></tr>
"""
        for r in fancy_bills:
            html += f"""        <tr class="fancy">
            <td>{r.get('position', '')}</td>
            <td>{r.get('serial', '')}</td>
            <td>{r.get('fancy_types', '')}</td>
            <td>{r.get('confidence', '')}</td>
        </tr>
"""

        html += """    </table>

    <h2>All Bills</h2>
    <table>
        <tr><th>Position</th><th>File</th><th>Serial</th><th>Patterns</th><th>Status</th></tr>
"""
        for r in self.current_results:
            css_class = ""
            if r.get('is_fancy'):
                css_class = "fancy"
            elif r.get('needs_review'):
                css_class = "review"

            status = []
            if r.get('is_fancy'):
                status.append("Fancy")
            if r.get('needs_review'):
                status.append("Review")
            if r.get('error'):
                status.append(r['error'])

            html += f"""        <tr class="{css_class}">
            <td>{r.get('position', '')}</td>
            <td>{r.get('front_file', '')}</td>
            <td>{r.get('serial', '')}</td>
            <td>{r.get('fancy_types', '')}</td>
            <td>{', '.join(status) if status else 'OK'}</td>
        </tr>
"""

        html += """    </table>
</body>
</html>"""
        return html

    def _start_processing(self, input_dir: str, output_dir: str):
        """Start the processing in a background thread."""
        from .processing_thread import ProcessingThread

        self.processing_thread = ProcessingThread(
            input_dir=input_dir,
            output_dir=output_dir,
            use_gpu=self.settings.processing.use_gpu,
            verify_pairs=self.settings.processing.verify_pairs,
            crop_all=self.settings.processing.crop_all
        )
        self.processing_thread.progress_updated.connect(self._on_progress_updated)
        self.processing_thread.result_ready.connect(self._on_result_ready)
        self.processing_thread.processing_complete.connect(self._on_processing_complete)
        self.processing_thread.error_occurred.connect(self._on_processing_error)
        self.processing_thread.start()

    @Slot(int, int, str)
    def _on_progress_updated(self, current: int, total: int, message: str):
        """Handle progress update from processing thread."""
        self.processing_panel.update_progress(current, total)
        self.progress_label.setText(f"{current}/{total}")
        self.status_label.setText(message)

    @Slot(dict)
    def _on_result_ready(self, result: dict):
        """Handle a single result from processing."""
        self.current_results.append(result)
        self.results_list.add_result(result)

    @Slot(dict)
    def _on_processing_complete(self, summary: dict):
        """Handle processing completion."""
        self.is_processing = False
        self.processing_panel.set_processing(False)

        total = summary.get('total', 0)
        fancy = summary.get('fancy_count', 0)
        review = summary.get('review_count', 0)

        self.status_label.setText(
            f"Complete: {total} bills processed, {fancy} fancy, {review} need review"
        )
        self.progress_label.setText("")

        # Auto-export if enabled
        if self.current_results:
            self._auto_export(summary)

    @Slot(str)
    def _on_processing_error(self, error: str):
        """Handle processing error."""
        self.is_processing = False
        self.processing_panel.set_processing(False)
        QMessageBox.critical(self, "Processing Error", error)
        self.status_label.setText(f"Error: {error}")

    def _apply_settings(self):
        """Apply changed settings to the UI."""
        # Update theme if needed
        pass  # Theme changes would go here


def run_gui():
    """Launch the GUI application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Dollar Bill Processor")
    app.setOrganizationName("DollarBillProcessor")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()
