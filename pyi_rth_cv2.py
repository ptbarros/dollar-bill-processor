"""PyInstaller runtime hook: make `import cv2` robust in the macOS .app bundle.

opencv-python's `cv2/__init__.py` runs a bootstrap that locates and loads the
native extension by manipulating sys.path. In a macOS `.app`, PyInstaller splits
the package across Contents/Frameworks and Contents/Resources, so the bootstrap
can't find the native `.so` next to its config and re-imports the package instead
-> "recursion is detected during loading of cv2 binary extensions".

This hook runs before any user code. When frozen, it finds the bundled cv2 native
extension (wherever PyInstaller placed it) and registers it as the top-level
`cv2` module directly, so the package bootstrap is skipped entirely. The native
module provides the OpenCV API the app and RapidOCR use (imread, cvtColor,
findContours, ...). It is a no-op off macOS, when not frozen, or if no extension
is found (in which case cv2's normal import path still runs).
"""
import os
import sys


def _preload_cv2():
    if not getattr(sys, "frozen", False):
        return
    if sys.platform != "darwin":
        return
    if "cv2" in sys.modules:
        return

    base = getattr(sys, "_MEIPASS", None)
    if not base:
        return

    # Locate the cv2 native extension anywhere under the bundle.
    ext = None
    for root, _dirs, files in os.walk(base):
        for f in files:
            if f.startswith("cv2") and (f.endswith(".so") or f.endswith(".pyd")):
                ext = os.path.join(root, f)
                break
        if ext:
            break
    if not ext:
        return  # let cv2's own import path try (and report) instead

    try:
        import numpy  # noqa: F401  (cv2's native module needs numpy present first)
        import importlib.util
        spec = importlib.util.spec_from_file_location("cv2", ext)
        module = importlib.util.module_from_spec(spec)
        sys.modules["cv2"] = module
        spec.loader.exec_module(module)
    except Exception:
        # Loading directly failed; drop our entry so the normal path can run.
        sys.modules.pop("cv2", None)


_preload_cv2()
