# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Dollar Detective (torch-free build).

onedir build. Bundles the ONNX model, patterns, config, and RapidOCR's model
data; excludes the heavy torch/ultralytics/easyocr stack (the app runs on
onnxruntime + rapidocr). Reusable on Windows (produces .exe) and Linux.
"""
import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

# --- app data files ---
datas = [
    ('best.onnx', '.'),
    ('best.pt', '.'),        # only needs to exist; ONNX sibling drives inference
    ('config.yaml', '.'),
    ('assets/icon.png', 'assets'),   # window / taskbar icon
]
# patterns/ tree, preserving structure (skip caches and the writable user dir --
# user patterns live in the per-user data dir, not the read-only bundle).
for p in Path('patterns').rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts and 'user' not in p.parts:
        datas.append((str(p), str(p.parent)))

binaries = []
hiddenimports = ['yaml', 'pandas', 'openpyxl', 'docx', 'PIL', 'updater', 'gui.updater_ui']

# Packages that ship data and/or dynamically load native libs / submodules:
#   - rapidocr_onnxruntime: ONNX models + config.yaml
#   - onnxruntime: provider shared libs
#   - lupa: dynamically imports a versioned native lib (lupa._lua54 etc.)
#   - anthropic / openai: AI pattern generation; imported lazily in the AI tab,
#     so they must be forced in. Pull compiled deps (pydantic_core, jiter) and
#     read their own version via package metadata -> collect_all grabs all three.
_collect_pkgs = ['rapidocr_onnxruntime', 'onnxruntime', 'lupa', 'anthropic', 'openai']

# cv2 (opencv): cv2 is imported at module level, so PyInstaller's built-in cv2
# hook already collects it correctly (it EXECs cv2's config to place the native
# extension at the right path). Adding an explicit collect_all('cv2') ON TOP puts
# a SECOND copy of the extension/.dylibs at a different location. Windows/Linux
# loaders tolerate that, but macOS's strict dyld can't resolve it and `import cv2`
# fails at runtime ("Missing required dependencies: opencv-python-headless"). So on
# macOS rely solely on the built-in hook; keep the belt-and-suspenders collect_all
# on Windows/Linux where it's proven and harmless.
if sys.platform != 'darwin':
    _collect_pkgs.append('cv2')

for pkg in _collect_pkgs:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# The whole point of the de-torch: keep these OUT of the bundle.
excludes = [
    'torch', 'torchvision', 'ultralytics', 'easyocr',
    'scipy', 'matplotlib', 'nvidia',  # only pulled in by the excluded stack
    'PyQt5', 'PyQt6',                  # app uses PySide6; QScintilla drags PyQt in
]

# Release builds have no console window (errors still go to the debug log in the
# user data dir). Set DBP_BUILD_CONSOLE=1 to keep a console for debugging.
_console = os.environ.get('DBP_BUILD_CONSOLE') == '1'

# macOS: a runtime hook loads cv2's native extension directly, sidestepping the
# opencv bootstrap that recurses in the split Frameworks/Resources .app layout.
_runtime_hooks = ['pyi_rth_cv2.py'] if sys.platform == 'darwin' else []

a = Analysis(
    ['run_gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=_runtime_hooks,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DollarDetective',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=_console,   # release: no console (DBP_BUILD_CONSOLE=1 to debug)
    icon='assets/icon.ico',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='DollarDetective',
)

# macOS: wrap the onedir into a proper .app bundle (Windows/Linux ignore this).
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='Dollar Detective.app',
        icon=None,   # .icns not generated yet; uses the default app icon
        bundle_identifier='com.paulbarros.dollardetective',
    )
