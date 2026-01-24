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


def main():
    """Main entry point."""
    if not check_dependencies():
        sys.exit(1)

    from gui.main_window import run_gui
    run_gui()


if __name__ == "__main__":
    main()
