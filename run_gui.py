#!/usr/bin/env python3
"""
Dollar Bill Processor - GUI Launcher

Launches the graphical user interface for processing and reviewing bills.
"""

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
    from resource_path import app_base
    from process_production import ProductionProcessor

    model = app_base() / "best.pt"
    proc = ProductionProcessor(str(model))
    print(f"SELFTEST backend: yolo_onnx={getattr(proc, 'use_onnx', '?')} ocr={proc.ocr_reader.name}")
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
