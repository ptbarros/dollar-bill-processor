# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Dollar Bill Processor (torch-free build).

onedir build. Bundles the ONNX model, patterns, config, and RapidOCR's model
data; excludes the heavy torch/ultralytics/easyocr stack (the app runs on
onnxruntime + rapidocr). Reusable on Windows (produces .exe) and Linux.
"""
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

# --- app data files ---
datas = [
    ('best.onnx', '.'),
    ('best.pt', '.'),        # only needs to exist; ONNX sibling drives inference
    ('config.yaml', '.'),
]
# patterns/ tree, preserving structure (skip caches and the writable user dir --
# user patterns live in the per-user data dir, not the read-only bundle).
for p in Path('patterns').rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts and 'user' not in p.parts:
        datas.append((str(p), str(p.parent)))

binaries = []
hiddenimports = ['yaml', 'pandas', 'openpyxl', 'docx', 'PIL']

# Packages that ship data and/or dynamically load native libs / submodules:
#   - rapidocr_onnxruntime: ONNX models + config.yaml
#   - onnxruntime: provider shared libs
#   - lupa: dynamically imports a versioned native lib (lupa._lua54 etc.)
for pkg in ('rapidocr_onnxruntime', 'onnxruntime', 'lupa'):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# The whole point of the de-torch: keep these OUT of the bundle.
excludes = [
    'torch', 'torchvision', 'ultralytics', 'easyocr',
    'scipy', 'matplotlib', 'nvidia',  # only pulled in by the excluded stack
]

a = Analysis(
    ['run_gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DollarBillProcessor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,   # keep a console for Phase 2 debugging; flip to False for release
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='DollarBillProcessor',
)
