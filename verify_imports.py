"""Shared frozen-build import self-check.

Enabled via DBP_VERIFY_IMPORTS=1 on a packaged build. CI runs the frozen app/exe
with it set so a bundle that can't load its native deps (notably cv2 on macOS) is
caught before the installer/dmg is published, and prints the real traceback.

Kept in its own tiny module (no heavy imports) so both the main app (run_gui) and
the standalone crop tool (crop_tool) can reuse it without dragging each other's
dependencies into their bundle.
"""
import sys


def verify_imports() -> int:
    import traceback
    try:
        from version import __version__
        print(f"Dollar Detective {__version__} — import verification")
    except Exception:
        pass
    mods = ["PySide6.QtWidgets", "cv2", "numpy", "onnxruntime",
            "rapidocr_onnxruntime", "yaml", "PIL"]
    ok = True
    for m in mods:
        try:
            __import__(m)
            print(f"  OK   {m}")
        except Exception as e:
            ok = False
            print(f"  FAIL {m}: {type(e).__name__}: {e}")
            traceback.print_exc()
            if m == "cv2":
                _dump_cv2_layout()
    print("VERIFY " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def _dump_cv2_layout():
    """List where cv2's files actually landed in a frozen bundle (diagnostics)."""
    base = getattr(sys, "_MEIPASS", None)
    if not base:
        return
    import os
    print(f"  --- cv2 layout under {base} ---")
    for root, _dirs, files in os.walk(base):
        for f in files:
            low = f.lower()
            if "cv2" in low or low.endswith(("config.py", "config-3.py")):
                print(f"      {os.path.join(root, f)}")
