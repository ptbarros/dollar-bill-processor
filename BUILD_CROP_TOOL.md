# Building the standalone Crop Tool (Windows .exe)

`crop_tool.py` is the eBay cropping feature packaged on its own, for distributing
to other collectors without shipping the full Dollar Detective app. This
builds a single self-contained `DollarBillCropTool.exe`.

## What it ships (and what it deliberately doesn't)

- **Model:** only an ONNX model, **renamed to `detector.bin`** (the spec copies it
  from `best.onnx` at build time). **No `best.pt`** — so the weights can't be
  retrained/fine-tuned, and there's no obvious `best.onnx`/`best.pt` to grab.
- **Onefile:** everything is packed inside the .exe (assets aren't loose on disk).
- **Excludes** torch / ultralytics / easyocr — the tool runs on onnxruntime +
  RapidOCR. This keeps the AGPL-licensed `ultralytics` library **out** of the
  distributable and shrinks the build.
- Size: ~300 MB (onnxruntime + PySide6 + OpenCV + RapidOCR models). Inherent to a
  self-contained ML app.

**Obscurity, not security:** a determined person can still unpack any PyInstaller
exe and recover the model. These steps stop casual copying, which is the intent.

## Easiest: build in GitHub Actions (no Windows machine needed)

The `crop-tool-windows` job in `.github/workflows/build.yml` builds the exe on a
`windows-latest` runner (OpenVINO edition), same as the full-app installers.

- **Trigger:** Actions tab → "Build installers" → **Run workflow** (workflow_dispatch),
  or push a `v*` tag (it also builds then).
- **Download:** open the finished run → **Artifacts** → `crop-tool-windows`
  (`DollarBillCropTool-<version>.exe`).
- It is uploaded as a **workflow artifact only** — deliberately **not** attached to
  the public GitHub Release, so you can hand it to FIL's friends privately without
  exposing the full app. To publish it in the Release later, add
  `artifacts/crop-tool-windows/*` to the `release` job's `files:` list and add
  `crop-tool-windows` to that job's `needs:`.

The rest of this doc is for building **locally** if you ever want to.

## Prerequisites (on the Windows build machine)

1. Python 3.12 (match the dev version).
2. A venv with the **torch-free** dependencies — the same set used to build the
   full app (`DollarDetective.spec`):
   ```
   pip install pyinstaller onnxruntime rapidocr_onnxruntime opencv-python-headless \
       PySide6 lupa pyyaml pandas openpyxl pillow numpy
   ```
   (Use `opencv-python` if the tool ever needs GUI imshow; headless is fine here.)
3. **`best.onnx` present in the repo root.** If you only have `best.pt`, export it
   first (ultralytics): `yolo export model=best.pt format=onnx opset=12`
   — do this in a *separate* env that has ultralytics/torch; the build env doesn't
   need them. The spec then copies `best.onnx` → `detector.bin` automatically.

## Build

From the repo root, in the build venv:
```
pyinstaller crop_tool.spec
```
Output: `dist\DollarBillCropTool.exe`.

- Keep a console window for debugging: `set DBP_BUILD_CONSOLE=1` before building.
- The exe icon uses `assets/icon.ico` if present (Windows only).

## First-run check

Launch the exe, set an **Input folder** (front+back scans), click **Crop settings…**
(loads the model + a sample bill for the preview), then **Run** with an output
folder. Confirm crops appear and no `results_*.csv` clutter is written to the input
folder (the tool passes `write_reports=False`).

## Notes / options

- **Onefile vs onedir:** onefile is one tidy .exe but unpacks to a temp dir on each
  launch (slower first paint). For faster startup, switch the spec to a `COLLECT`
  (onedir) build like `DollarDetective.spec` — you then distribute a folder.
- **Installer:** to wrap the exe in a Windows installer, adapt
  `installer/DollarDetective.iss` (Inno Setup) to point at `DollarBillCropTool`.
- **Model updates:** re-export `best.onnx` and rebuild — the spec re-copies
  `detector.bin` when `best.onnx` is newer.
- **Stronger protection (later):** Tier 2 would encrypt `detector.bin` and decrypt
  it in memory (`onnxruntime.InferenceSession(bytes)`), so the plaintext model never
  hits disk. Not implemented yet.
