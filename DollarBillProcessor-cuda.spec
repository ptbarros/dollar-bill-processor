# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the CUDA edition (NVIDIA GPU, torch + EasyOCR).

Unlike the default torch-free build (DollarBillProcessor.spec), this bundles the
full torch/ultralytics/easyocr stack so YOLO detection and OCR both run on the
CUDA GPU -- the fast path proven on the RTX box (~33s/100 bills). A runtime hook
forces DBP_FORCE_TORCH=1 + DBP_OCR=easy. Large bundle (CUDA libs come with torch).
Windows-only (that's where the CUDA wheels + the RTX box are).
"""
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

# --- app data files ---
datas = [
    ('best.pt', '.'),        # torch YOLO model drives inference in this edition
    ('config.yaml', '.'),
    ('assets/icon.png', 'assets'),   # window / taskbar icon
]
# NOTE: best.onnx is deliberately NOT bundled so load_detector uses the torch
# path even if the runtime hook were bypassed.
for p in Path('patterns').rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts and 'user' not in p.parts:
        datas.append((str(p), str(p.parent)))

binaries = []
hiddenimports = ['yaml', 'pandas', 'openpyxl', 'docx', 'PIL',
                 'scipy', 'skimage', 'torchvision']

# Collect packages that ship data and/or dynamically load native libs:
#   - torch / torchvision: CUDA runtime DLLs live in torch/lib
#   - ultralytics: YOLO model plumbing + assets
#   - easyocr: recognition/detection modules (weights auto-download on first run)
#   - lupa: versioned native lua lib
#   - anthropic / openai: AI pattern generation (lazily imported in the AI tab);
#     collect_all pulls their compiled deps (pydantic_core, jiter) + metadata
for pkg in ('torch', 'torchvision', 'ultralytics', 'easyocr', 'lupa', 'anthropic', 'openai'):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Keep only the conflicting Qt bindings out; torch/easyocr deps must stay in.
excludes = ['PyQt5', 'PyQt6']

_console = os.environ.get('DBP_BUILD_CONSOLE') == '1'

a = Analysis(
    ['run_gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['rthook_force_cuda.py'],
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
    console=_console,
    icon='assets/icon.ico',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='DollarBillProcessor',
)
