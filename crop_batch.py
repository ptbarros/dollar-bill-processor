"""Shared lean crop-only batch worker.

Runs the real pipeline's cropping over a folder of bills WITHOUT the full
classification path (no pattern matching, seal-shift / gas-pump, plate reads, or
results CSV -- none of which a crop run needs). Used by both the standalone
crop tool (``crop_tool.py``) and the main app's Crop Manager ("Run on Folder…"),
so a batch produces exactly the crops the Crop Manager is configured to make.

OCR runs only when ``read_serial`` is set (to name crops by serial number).
"""

from pathlib import Path

from PySide6.QtCore import QThread, Signal

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


class CropWorker(QThread):
    progress = Signal(int, int, str)     # index, total, label
    log = Signal(str)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, input_dir: Path, output_dir: Path, use_gpu: bool,
                 read_serial: bool, config_path: Path = None):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.use_gpu = use_gpu
        self.read_serial = read_serial
        # When set, crop with this config file instead of the saved one -- lets
        # the Crop Manager run a batch using its in-dialog (unsaved) edits.
        self.config_path = config_path

    def run(self):
        try:
            from process_production import ProductionProcessor, Config, ScannerFormatDetector

            base = app_base()
            yolo_path, onnx_path = _resolve_model()
            if not ((onnx_path and onnx_path.exists()) or yolo_path.exists()):
                self.failed.emit("Detection model not found in the app folder.")
                return

            cfg_path = Path(self.config_path) if self.config_path else _active_config_path()
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
            # crop run uses. OCR runs ONLY when serial-in-filename is enabled.
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
                if self.isInterruptionRequested():
                    self.log.emit("Cancelled.")
                    break
                self.progress.emit(i, total, pair.front_path.name)
                if getattr(pair, 'error', None):
                    continue
                try:
                    if self.read_serial:
                        serial, _conf, is_ud, _bvar, star, align_info = processor.extract_serial(
                            pair.front_path, cached_detections=pair.front_cache)
                        if serial and star and not serial.endswith('*'):
                            serial = serial[:-1] + '*' if len(serial) == 10 else serial + '*'
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
