#!/usr/bin/env python3
"""
Dollar Detective - GUI Launcher

Launches the graphical user interface for processing and reviewing bills.
"""

import os
import sys
from pathlib import Path

# Ensure we can import from the gui package
sys.path.insert(0, str(Path(__file__).parent))


def check_dependencies():
    """Check that required dependencies are installed."""
    missing = []

    try:
        import PySide6
    except ImportError:
        missing.append("PySide6")

    try:
        import cv2
    except ImportError:
        missing.append("opencv-python-headless")

    try:
        import yaml
    except ImportError:
        missing.append("pyyaml")

    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False

    return True


def _selftest(image_path: str) -> int:
    """Headless smoke test: load the model + OCR and read a serial from an image.

    Enabled via DBP_SELFTEST=<image_path>. Used to verify a packaged (frozen)
    build can do real ONNX + RapidOCR inference without a display. Prints PASS/FAIL.
    """
    from pathlib import Path
    import cv2
    from resource_path import app_base, user_data_dir
    from process_production import ProductionProcessor

    # Writable user-data check: the read-only bundle must not be written to.
    from settings_manager import get_settings
    try:
        settings = get_settings()
        settings.save()  # the call that failed inside the read-only bundle
        print(f"SELFTEST settings: writable -> {settings.path}")
    except Exception as e:
        print(f"SELFTEST FAIL: settings.save() -> {type(e).__name__}: {e}")
        return 1

    model = app_base() / "best.pt"
    # DBP_SELFTEST_GPU=1 exercises the accelerated provider path (OpenVINO / CUDA /
    # DirectML) so a packaged build can be verified end-to-end, not just on CPU.
    use_gpu = os.environ.get("DBP_SELFTEST_GPU") == "1"
    proc = ProductionProcessor(str(model), use_gpu=use_gpu)
    providers = getattr(getattr(proc, "yolo_model", None), "providers", None)
    print(f"SELFTEST backend: yolo_onnx={getattr(proc, 'use_onnx', '?')} "
          f"ocr={proc.ocr_reader.name} use_gpu={use_gpu} providers={providers}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"SELFTEST FAIL: could not read {image_path}")
        return 1
    dets = proc._detect_all_objects(img, conf=0.3)
    n_serial = len(dets.get("serial_number", []))
    serial, conf = proc.extract_serial(Path(image_path))[:2]
    ok = bool(n_serial) and bool(serial)
    print(f"SELFTEST {'PASS' if ok else 'FAIL'}: serial_regions={n_serial} serial={serial!r} conf={conf:.2f}")
    return 0 if ok else 1


def main():
    """Main entry point."""
    import os
    selftest_img = os.environ.get("DBP_SELFTEST")
    if selftest_img:
        sys.exit(_selftest(selftest_img))

    if not check_dependencies():
        sys.exit(1)

    from gui.main_window import run_gui
    run_gui()


if __name__ == "__main__":
    main()
