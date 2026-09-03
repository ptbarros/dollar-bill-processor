# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone Crop Tool (onefile, torch-free).

Produces a single self-contained executable of crop_tool.py for distributing the
eBay cropping feature on its own — without the full Dollar Bill Processor app.

Model obscurity (Tier 0+1):
  - Ships ONLY an ONNX model, renamed to ``detector.bin`` (this spec copies it
    from best.onnx at build time). No ``best.pt`` -> the weights can't be
    retrained/fine-tuned, and there's no obvious "best.pt"/"best.onnx" to grab.
  - Onefile: assets are packed into the exe, not loose on disk.
  - Excludes torch/ultralytics/easyocr so the AGPL ultralytics library is NOT
    shipped and the build stays small (~150-250 MB vs ~500 MB).
None of this is unbreakable — a determined person can still unpack a PyInstaller
exe — but it stops casual copying, which is the intent.

Build (from the repo root, in the torch-free build venv):
    pyinstaller crop_tool.spec            # -> dist/DollarBillCropTool(.exe)
Set DBP_BUILD_CONSOLE=1 to keep a console window for debugging.
"""
import os
import shutil
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

# --- Prepare the obscured model: best.onnx -> detector.bin (inference-only) ---
_src_onnx = Path('best.onnx')
_obf = Path('detector.bin')
if not _src_onnx.exists():
    raise SystemExit("crop_tool.spec: best.onnx not found — export it before building.")
if not _obf.exists() or _obf.stat().st_mtime < _src_onnx.stat().st_mtime:
    shutil.copyfile(_src_onnx, _obf)

# --- app data files (NO best.pt / best.onnx — only the obscured model) ---
datas = [
    ('detector.bin', '.'),
    ('config.yaml', '.'),
]
# patterns/ tree (needed by the pattern engine for serial-based crop naming),
# skipping caches and the writable user dir.
for p in Path('patterns').rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts and 'user' not in p.parts:
        datas.append((str(p), str(p.parent)))

binaries = []
hiddenimports = [
    'yaml', 'pandas', 'openpyxl', 'PIL',
    # lazily imported inside crop_tool / main paths:
    'gui.crop_dialog', 'crop_preview', 'process_production', 'resource_path',
]

# Packages that ship data and/or dynamically load native libs / submodules.
for pkg in ('rapidocr_onnxruntime', 'onnxruntime', 'lupa'):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Keep the heavy / AGPL stack OUT (crop tool runs on onnxruntime + rapidocr).
excludes = [
    'torch', 'torchvision', 'ultralytics', 'easyocr',
    'scipy', 'matplotlib', 'nvidia',
    'PyQt5', 'PyQt6',
]

_console = os.environ.get('DBP_BUILD_CONSOLE') == '1'
_icon = 'assets/icon.ico' if Path('assets/icon.ico').exists() else None

a = Analysis(
    ['crop_tool.py'],
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

# Onefile: bundle binaries + datas straight into the EXE (no COLLECT).
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='DollarBillCropTool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=_console,
    icon=_icon,
)
