#!/usr/bin/env python3
"""
Standalone Crop Tool — crop a folder of bills with the eBay Crop Manager settings.

Purpose: a small, self-contained GUI for cropping a handful of individually
flat-bed-scanned bills (e.g. rare notes that don't go through the feed scanner)
WITHOUT opening the full processing app. Point it at an input folder and an
output folder, hit Run, and it aligns + crops every bill using the exact same
crop settings the main app's "eBay Crop Manager" edits (config.yaml).

It reuses the real pipeline:
  ProductionProcessor.process_directory(..., crop_all=True)
and the real settings dialog:
  gui.crop_dialog.EbayCropDialog

so crops are identical to what the main app produces. Input is expected to be
front + back scans of each bill (pairing + front/back detection is automatic).

Run:  python crop_tool.py
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QProgressBar, QCheckBox, QMessageBox,
    QPlainTextEdit, QGroupBox, QComboBox,
)
from PySide6.QtCore import Qt, QThread, Signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resource_path import app_base, user_data_dir


def _active_config_path() -> Path:
    """The config the main app reads/writes: user copy if present, else bundled."""
    user_cfg = user_data_dir() / "config.yaml"
    return user_cfg if user_cfg.exists() else app_base() / "config.yaml"


def _resolve_model():
    """Return (yolo_model_path, onnx_model_path) for the detector.

    Packaged builds ship only an obscured ONNX model named ``detector.bin`` (no
    ``best.pt``, so it can't be retrained). Dev runs use ``best.onnx``/``best.pt``.
    onnx_model_path drives inference; yolo_model_path is only the torch fallback.
    """
    base = app_base()
    obf = base / "detector.bin"
    if obf.exists():
        return obf, obf
    return base / "best.pt", (base / "best.onnx" if (base / "best.onnx").exists() else None)


# =============================================================================
# Worker thread: runs the real pipeline off the UI thread
# =============================================================================
class CropWorker(QThread):
    progress = Signal(int, int, str)     # index, total, label
    log = Signal(str)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, input_dir: Path, output_dir: Path, use_gpu: bool, read_serial: bool):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.use_gpu = use_gpu
        self.read_serial = read_serial

    def run(self):
        try:
            from process_production import ProductionProcessor, Config, ScannerFormatDetector

            base = app_base()
            yolo_path, onnx_path = _resolve_model()
            if not ((onnx_path and onnx_path.exists()) or yolo_path.exists()):
                self.failed.emit("Detection model not found in the app folder.")
                return

            cfg_path = _active_config_path()
            cfg = Config(cfg_path if cfg_path.exists() else None, None)
            patterns_dir = base / "patterns"

            self.log.emit(f"Loading model + settings ({cfg_path.name})…")
            processor = ProductionProcessor(
                yolo_path,
                use_gpu=self.use_gpu,
                cfg=cfg,
                patterns_dir=patterns_dir if patterns_dir.exists() else None,
                onnx_model_path=onnx_path,
            )

            # Lean crop-only path: find pairs, verify front/back (YOLO), align +
            # crop. Deliberately does NOT classify patterns, compute seal-shift /
            # gas-pump, read plates, or write a results CSV — none of which the
            # crop tool uses. OCR runs ONLY when serial-in-filename is enabled.
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.log.emit(f"Scanning {self.input_dir} …")
            _fmt, pairs = ScannerFormatDetector.find_pairs(self.input_dir)
            if not pairs:
                self.failed.emit("No bill pairs found in the input folder.")
                return
            pairs = processor.verify_and_swap_pairs(pairs)

            total = len(pairs)
            self.log.emit(f"Cropping {total} bills"
                          + (" (reading serials)…" if self.read_serial else "…"))
            cropped = 0
            for i, pair in enumerate(pairs, 1):
                self.progress.emit(i, total, pair.front_path.name)
                if getattr(pair, 'error', None):
                    continue
                try:
                    if self.read_serial:
                        serial, _conf, is_ud, _bvar, star, align_info = processor.extract_serial(
                            pair.front_path, cached_detections=pair.front_cache)
                        if serial and star and not serial.endswith('*'):
                            _tlen = processor._serial_format()[3]
                            serial = serial[:-1] + '*' if len(serial) == _tlen else serial + '*'
                        pair.serial = serial or pair.front_path.stem
                        pair.is_upside_down = is_ud
                        pair.front_align_angle = align_info.get('angle', 0.0)
                        pair.front_align_flipped = align_info.get('flipped', False)
                        processor.generate_crops(pair, self.output_dir)
                    else:
                        pair.serial = None
                        processor.generate_crops(pair, self.output_dir,
                                                 name=pair.front_path.stem)
                    cropped += 1
                except Exception as e:
                    self.log.emit(f"  skipped {pair.front_path.name}: {e}")
            self.done.emit({'total': total, 'cropped': cropped})
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


# =============================================================================
# Main window
# =============================================================================
class CropToolWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dollar Bill Crop Tool")
        self.setMinimumWidth(560)
        self.worker: CropWorker | None = None
        icon = app_base() / "assets" / "DD-Crop.png"
        if icon.exists():
            from PySide6.QtGui import QIcon
            self.setWindowIcon(QIcon(str(icon)))
        self._build_ui()

    def _build_ui(self):
        v = QVBoxLayout(self)

        intro = QLabel(
            "Crop a folder of scanned bills using your Crop Manager settings.\n"
            "Meant for individually scanned bills that skip the feed scanner.")
        intro.setWordWrap(True)
        v.addWidget(intro)

        folders = QGroupBox("Folders")
        g = QGridLayout(folders)
        g.addWidget(QLabel("Input folder:"), 0, 0)
        self.in_edit = QLineEdit()
        g.addWidget(self.in_edit, 0, 1)
        in_btn = QPushButton("Browse…"); in_btn.clicked.connect(lambda: self._browse(self.in_edit))
        g.addWidget(in_btn, 0, 2)
        g.addWidget(QLabel("Output folder:"), 1, 0)
        self.out_edit = QLineEdit()
        g.addWidget(self.out_edit, 1, 1)
        out_btn = QPushButton("Browse…"); out_btn.clicked.connect(lambda: self._browse(self.out_edit))
        g.addWidget(out_btn, 1, 2)
        v.addWidget(folders)

        # Crop profile: pick which saved crop setup to use (e.g. $1 vs $5). Edit
        # the profiles in Crop settings.
        prof = QHBoxLayout()
        prof.addWidget(QLabel("Crop profile:"))
        self.profile_combo = QComboBox()
        self.profile_combo.setToolTip("Which saved crop setup to use. Manage these in Crop settings…")
        self.profile_combo.currentTextChanged.connect(self._on_profile_selected)
        prof.addWidget(self.profile_combo, 1)
        prof.addStretch()
        v.addLayout(prof)

        opts = QHBoxLayout()
        # This tool always crops every bill in the folder (that's the whole point
        # of opening it separately), so there's no "crop all" toggle.
        self.serial_check = QCheckBox("Include serial number in filename")
        self.serial_check.setChecked(True)
        self.serial_check.setToolTip(
            "On: read each bill's serial (via OCR) and name crops by it — handy for "
            "organizing.\nOff: name crops by the source file (no OCR, a bit faster).")
        opts.addWidget(self.serial_check)
        self.gpu_check = QCheckBox("Use GPU")
        self.gpu_check.setChecked(True)
        self.gpu_check.setToolTip("Use GPU acceleration if available.")
        opts.addWidget(self.gpu_check)
        opts.addStretch()
        settings_btn = QPushButton("Crop settings…")
        settings_btn.setToolTip("Edit which crops are generated (same dialog as the main app).")
        settings_btn.clicked.connect(self._open_crop_settings)
        opts.addWidget(settings_btn)
        v.addLayout(opts)

        self.run_btn = QPushButton("Run")
        self.run_btn.setMinimumHeight(38)
        self.run_btn.clicked.connect(self._run)
        v.addWidget(self.run_btn)

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        v.addWidget(self.progress)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        v.addWidget(self.log)

        self._refresh_profiles()

    # --- crop profiles ----------------------------------------------------
    def _read_config(self) -> dict:
        import yaml
        p = _active_config_path()
        if p.exists():
            try:
                return yaml.safe_load(open(p)) or {}
            except Exception:
                return {}
        return {}

    def _refresh_profiles(self):
        """Populate the profile dropdown from the active config."""
        cfg = self._read_config()
        profiles = cfg.get('crop_profiles') or {}
        names = list(profiles.keys()) or ['Default']
        active = cfg.get('active_crop_profile')
        if active not in names:
            active = names[0]
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItems(names)
        self.profile_combo.setCurrentText(active)
        self.profile_combo.setEnabled(len(names) > 1 or bool(profiles))
        self.profile_combo.blockSignals(False)

    def _on_profile_selected(self, name):
        """Persist the chosen active profile so the run uses it."""
        if not name:
            return
        import yaml
        cfg = self._read_config()
        if not cfg.get('crop_profiles'):
            return   # nothing to switch until profiles are saved in Crop settings
        cfg['active_crop_profile'] = name
        user_cfg = user_data_dir() / "config.yaml"
        user_cfg.parent.mkdir(parents=True, exist_ok=True)
        with open(user_cfg, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        self._append_log(f"Active crop profile: {name}")

    # --- helpers ----------------------------------------------------------
    def _browse(self, edit: QLineEdit):
        start = edit.text() or str(Path.home())
        d = QFileDialog.getExistingDirectory(self, "Choose folder", start)
        if d:
            edit.setText(d)

    def _get_preview_processor(self):
        """Lazily build (and cache) a ProductionProcessor for crop previews."""
        if getattr(self, "_preview_processor", None) is None:
            from process_production import ProductionProcessor, Config
            base = app_base()
            cfg_path = _active_config_path()
            cfg = Config(cfg_path if cfg_path.exists() else None, None)
            patterns = base / "patterns"
            yolo_path, onnx_path = _resolve_model()
            self._preview_processor = ProductionProcessor(
                yolo_path, use_gpu=False, cfg=cfg,
                patterns_dir=patterns if patterns.exists() else None,
                onnx_model_path=onnx_path)
        return self._preview_processor

    def _build_preview_ctx(self):
        """Build a crop preview context from the first bill in the input folder."""
        in_dir = self.in_edit.text().strip()
        if not in_dir or not Path(in_dir).is_dir():
            return None
        from PySide6.QtWidgets import QApplication
        from crop_preview import build_context_from_folder
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._append_log("Preparing crop preview (loading model + sample bill)…")
        QApplication.processEvents()
        try:
            return build_context_from_folder(self._get_preview_processor(), Path(in_dir))
        except Exception as e:
            self._append_log(f"Preview unavailable: {e}")
            return None
        finally:
            QApplication.restoreOverrideCursor()

    def _open_crop_settings(self):
        """Open the real eBay Crop Manager dialog on the active config.yaml."""
        import yaml
        from gui.crop_dialog import EbayCropDialog

        load_path = _active_config_path()
        config = {}
        if load_path.exists():
            with open(load_path) as f:
                config = yaml.safe_load(f) or {}

        preview_ctx = self._build_preview_ctx()
        if preview_ctx is None:
            self._append_log("Tip: set a valid input folder to preview crops on a sample bill.")
        dialog = EbayCropDialog(config, self, preview_ctx=preview_ctx)
        if dialog.exec():
            user_cfg = user_data_dir() / "config.yaml"
            user_cfg.parent.mkdir(parents=True, exist_ok=True)
            with open(user_cfg, "w") as f:
                yaml.dump(dialog.get_config(), f, default_flow_style=False, sort_keys=False)
            self._append_log(f"Saved crop settings to {user_cfg}")
            self._refresh_profiles()   # profiles / active may have changed

    def _append_log(self, msg: str):
        self.log.appendPlainText(msg)

    # --- run --------------------------------------------------------------
    def _run(self):
        input_dir = Path(self.in_edit.text().strip())
        output_dir = Path(self.out_edit.text().strip())
        if not input_dir.is_dir():
            QMessageBox.warning(self, "Input folder", "Please choose a valid input folder.")
            return
        if not self.out_edit.text().strip():
            QMessageBox.warning(self, "Output folder", "Please choose an output folder.")
            return

        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        self.log.clear()
        self._append_log("Starting…")

        self.worker = CropWorker(
            input_dir, output_dir,
            use_gpu=self.gpu_check.isChecked(),
            read_serial=self.serial_check.isChecked(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._append_log)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, i, total, name):
        self.progress.setMaximum(total)
        self.progress.setValue(i)
        self.progress.setFormat(f"%v / %m  —  {name}")

    def _on_done(self, results):
        self.progress.setValue(self.progress.maximum())
        out = self.out_edit.text().strip()
        self._append_log(f"Done. Crops written to {out}")
        self.run_btn.setEnabled(True)
        QMessageBox.information(self, "Done", f"Cropping complete.\nCrops saved to:\n{out}")

    def _on_failed(self, msg):
        self._append_log("ERROR:\n" + msg)
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", msg.splitlines()[0] if msg else "Unknown error")


def main():
    app = QApplication(sys.argv)
    win = CropToolWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
