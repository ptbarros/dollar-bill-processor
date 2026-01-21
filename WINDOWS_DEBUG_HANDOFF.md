# Windows Debug Handoff - Output Directory Not Created

## The Issue
When running `run_processor.bat` (by dragging a folder onto it), the script:
- ✅ Processes bills correctly
- ✅ Finds fancy serial numbers (4 BOOKENDS in canon folder)
- ❌ Does NOT create the `fancy_bills` output directory
- ❌ No crop files are generated

## Relevant Files
```
process_production.py   # Main Python script (line 692 creates output dir)
run_processor.bat       # Windows batch launcher
config.yaml             # Settings
patterns.txt            # Custom patterns
```

## What We've Tried
1. Added trailing backslash removal in batch file
2. Added explicit `mkdir` in batch file before Python runs
3. Neither fixed the issue

## Key Code Locations

**Batch file output path construction (run_processor.bat:49-52):**
```batch
:set_output
if "%INPUT_DIR:~-1%"=="\" set "INPUT_DIR=%INPUT_DIR:~0,-1%"
set "OUTPUT_DIR=%INPUT_DIR%\fancy_bills"
```

**Python directory creation (process_production.py:692):**
```python
output_dir.mkdir(parents=True, exist_ok=True)
```

**Python crop generation (process_production.py:631-656):**
```python
def generate_crops(self, pair: BillPair, output_dir: Path) -> list[Path]:
    # ...
    cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
```

## Debug Steps to Try

1. **Check what paths Python receives:**
   Add debug print in `process_production.py` around line 692:
   ```python
   print(f"DEBUG: output_dir = {output_dir}")
   print(f"DEBUG: output_dir.exists() = {output_dir.exists()}")
   output_dir.mkdir(parents=True, exist_ok=True)
   print(f"DEBUG: after mkdir, exists = {output_dir.exists()}")
   ```

2. **Check if generate_crops is called:**
   Add print in `process_production.py` around line 735:
   ```python
   if pair.is_fancy:
       fancy_bills.append(pair)
       print(f"DEBUG: Generating crops for {pair.serial} to {output_dir}")
       self.generate_crops(pair, output_dir)
   ```

3. **Check batch file variables:**
   Add after line 106 in `run_processor.bat`:
   ```batch
   echo DEBUG: INPUT_DIR=%INPUT_DIR%
   echo DEBUG: OUTPUT_DIR=%OUTPUT_DIR%
   dir "%OUTPUT_DIR%" 2>nul || echo DEBUG: Output dir does not exist yet
   pause
   ```

## Test Command
```batch
:: Drag canon folder onto run_processor.bat
:: Or run manually:
run_processor.bat "C:\path\to\canon"
```

## Expected Behavior
- Should create `canon\fancy_bills\` directory
- Should contain 40 crop files (4 bills × 10 crops each):
  - `B09206460C_01.jpg` through `B09206460C_10.jpg`
  - `B54215825E_01.jpg` through `B54215825E_10.jpg`
  - `F37335353T_01.jpg` through `F37335353T_10.jpg`
  - `L65525606Q_01.jpg` through `L65525606Q_10.jpg`
