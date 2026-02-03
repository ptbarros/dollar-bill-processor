# Dollar Bill Processor - Project Memory

## Overview
A GUI application for processing dollar bill images, detecting serial numbers via OCR, and classifying them against collectible "fancy serial number" patterns.

## Lua Pattern Plugin System (Added Feb 2025)

### Architecture
- **pattern_sandbox.py**: Secure Lua execution environment using `lupa` library
  - Whitelists safe functions (string, math, table, pairs, etc.)
  - Blocks dangerous functions (os, io, require, debug)
  - Instruction limits to prevent infinite loops

- **pattern_engine_v3.py**: Hybrid engine supporting both YAML and Lua patterns
  - Wraps v2 engine for backward compatibility
  - **Lua patterns fully override YAML** - if a Lua file exists for a pattern name, YAML is skipped entirely (even if Lua returns `matched = false`)
  - This prevents loose YAML patterns from matching when stricter Lua logic correctly rejects
  - Properties for backward compat: `config`, `config_path`, `user_config`, `user_config_path`, `patterns`

- **patterns/** directory structure:
  ```
  patterns/
  ├── core/           # Built-in Lua patterns (121 patterns)
  ├── user/           # User-created patterns (not in git)
  └── lib/
      └── helpers.lua # Shared utility functions (~400 lines)
  ```

### Lua Script Structure
```lua
--[[
Pattern: PATTERN_NAME
Description: What it matches
Tier: 1-10
Examples: ["12345678"]
--]]

function match(ctx)
    -- ctx.digits: "12345678" (8 numeric characters)
    -- ctx.full_serial: "A12345678B" (with prefix/suffix)
    -- ctx.digit_list: {1,2,3,4,5,6,7,8} as integers

    if not_matching then
        return {matched = false}
    end

    return {
        matched = true,
        highlights = {{positions = {0, 1}, color = "orange"}},
        connectors = {{from = 0, to = 7, color = "orange", style = "arc"}},
        group_boxes = {{from = 0, to = 2, color = "gold", thickness = 3}},
        message = "Description of match"
    }
end
```

### Visualization Features
- **highlights**: Color individual digit positions (each digit gets its own box)
- **connectors**: Lines between digit pairs (styles: arc, line, bracket, arrow, dashed)
- **group_boxes**: Single box spanning multiple digits (preferred for multi-digit groups)

### Visualization Best Practices
- Use `group_boxes` instead of `highlights` when you want one box around multiple digits (e.g., bookends, repeating groups)
- Use one arc per group, not one per digit (cleaner visual)
- For symmetric patterns (radar, pyramid), use arcs to show the mirror relationship
- Color coding: lime for ascending, cyan for descending, yellow for peaks, orange for bookends
- Example pyramid_ladder visualization: lime (ascending) → yellow (peak) → cyan (descending) with arcs connecting mirror positions

### Color Palette
purple, blue, cyan, orange, coral, gold, salmon, magenta, yellow, lime, teal, red, gray

### GUI Features
- Pattern Manager shows `[Lua]` indicator in purple for Lua-implemented patterns
- "View Script" button shows source code with Copy to Clipboard
- Script editor tab with syntax highlighting, templates, and live preview
- API Docs tab with copyable documentation for AI prompt generation
- "Re-classify All" button in results list
- Right-click "Re-classify" option for selected rows

### Key Files Modified
- `gui/pattern_dialog.py`: Extended CustomPatternDialog with script editor tabs
- `gui/preview_panel.py`: Added group_boxes rendering, Re-classify button
- `gui/results_list.py`: Added Re-classify All button and context menu option
- `process_production.py`: Updated to use v3 engine
- `requirements.txt`: Added `lupa>=2.0`

### Example User Patterns Created
- **MINIR**: Copy of MINI_REPEATER (3-digit repeat like 94680680)
- **TRPBK**: First 3 digits match last 3 digits (ABCxxABC like 61858618)

## Testing Commands
```bash
# Test pattern engine
python pattern_engine_v3.py

# Test sandbox security
python pattern_sandbox.py

# Test specific pattern
python -c "
from pattern_engine_v3 import PatternEngineV3
engine = PatternEngineV3()
matches = engine.classify_simple('A12344321B')
print(matches)
"

# Verify Lua pattern conversions
python tests/verify_lua_patterns.py
```

## Helper Functions (patterns/lib/helpers.lua)

### Core Analysis Functions
- `count_digits(s)` - Count occurrences of each digit, returns table
- `find_runs(s)` - Find consecutive runs, returns list of {digit, start, length}
- `unique_count(s)` - Count unique digits in string
- `digit_sum(s)` - Sum of all digits
- `most_common(s)` - Get most common digit and its count
- `get_unique_digits(s)` - Get sorted unique digits as string

### Pattern Detection
- `is_ladder(s)` / `is_ascending(s)` / `is_descending(s)` - Ladder checks
- `find_ladder_of_length(s, min_length)` - Find ladder of given length
- `find_longest_ladder(s)` - Find the longest ladder in string
- `is_palindrome(s)` - Check if string is palindrome
- `is_broken_palindrome(s, max_mismatches)` - Check for near-palindrome
- `is_repeater(s)` - Check ABCDABCD pattern
- `is_super_repeater(s)` - Check ABABABAB pattern
- `is_alternating(s)` - Check XYXYXYXY pattern
- `has_n_consecutive(s, n)` - Check for N consecutive identical digits

### Flipper Functions
- `all_flip_valid(s)` - Check if all digits are flip-valid (0,1,6,8,9)
- `flip_string(s)` - Get 180-degree rotated version

### Pair/Group Detection
- `find_pairs(s)` - Find consecutive pairs
- `find_consecutive_pairs(s)` - Find consecutive pairs with positions
- `has_four_consecutive_pairs(s)` - Check AABBCCDD pattern
- `has_three_consecutive_pairs_start(s)` - Check AABBCC at start
- `count_pairs(s)` - Count total pairs
- `find_triples(s)` - Find triple runs
- `find_quads(s)` - Find quad+ runs

### String Utilities
- `starts_with(s, prefix)` / `ends_with(s, suffix)` - String prefix/suffix checks
- `contains(s, substr)` - Substring check
- `only_digits(s, allowed)` - Check if string contains only specified digits
- `is_bookended(s, n)` - Check if first N and last N digits match

### Visualization Helpers
- `highlight(positions, color, label)` - Build highlight entry
- `highlight_range(start, stop, color, label)` - Highlight range
- `connector(from, to, color, style)` - Build connector entry
- `find_digit_positions(s, digit)` - Get all positions of a digit
- `find_matching_positions(s, digit_set)` - Get positions matching a set

### Counting Pattern Helpers
- `is_counting_pairs(s, step)` - Check two-digit pair counting pattern
- `check_counting_ladder(s)` - Check X0Y0Z0W0 pattern

## Notes
- User patterns in `patterns/user/` are gitignored (except .gitkeep)
- The v3 engine automatically loads helpers.lua into the sandbox
- Patterns are reloaded when `engine.reload()` is called
- All 117 YAML patterns have been converted to Lua (Feb 2025)
- Lua patterns provide richer visualization with custom highlights and connectors
- YAML patterns remain for metadata (odds, prices) but Lua fully controls detection
- Some YAML built-in checks (like `pyramid_ladder`) were too loose - Lua versions are stricter and more accurate
- The CSV output reflects whatever the pattern engine returns (Lua if file exists, otherwise YAML)

## Pattern-Specific Notes

### PYRAMID_LADDER
- True pyramid: digits go up by 1, then down by 1 (e.g., 12321, 1234321)
- Visualization: lime (ascending), yellow (peak), cyan (descending), orange arcs connecting mirror positions
- The old YAML check matched any "bounce" pattern regardless of step size - Lua is stricter

### Bookend Patterns (BOOKENDS, TRIPLE_BOOKENDS, DOUBLES_BOOKEND, TRIPLES_BOOKEND)
- Use `group_boxes` to wrap each bookend group in a single box
- Use one arc connecting the groups (not multiple arcs per digit)

## Plate Info Extraction & Mule Detection (Added Feb 2025)

### Overview
Optional feature to extract additional bill metadata and detect potential mule notes.

### Settings
- **Settings → Processing → "Extract plate and series info"**: Enable/disable (default: off)
- When enabled, extracts series_year, front_plate, and back_plate using YOLO + OCR
- Adds processing time due to additional OCR calls on detected regions

### Extracted Fields
| Field | YOLO Class | Location | Example |
|-------|------------|----------|---------|
| series_year | 8 | Front | "2013", "2017A" |
| front_plate | 4 | Front | "FW I 30", "G144" |
| back_plate | 0 | Back | "108", "523" |

### OCR Improvements (Feb 2025)
All plate info OCR uses **2x upscaling** before text recognition:
- **Series year**: Captures suffix letter (A, B, etc.) that appears below the year
- **Front plate**: Improves detection of "FW" and other small text
- **Back plate**: Reads both large and small font plates reliably

The `ocr_region()` helper function accepts an `upscale` parameter for this purpose.

### Mule Note Detection
A **mule note** is a bill with mismatched front/back plates from different printing eras:
- **Pre-1988**: Back plates had small font numbers
- **1988+**: Back plates transitioned to large font numbers
- **Mule**: A 1988+ series bill with an old-style small font back plate

**Detection logic:**
1. Check if series year is 1988 or later (large font era)
2. Check YOLO back_plate box height: ≤14px = small font, >14px = large font
3. If large font era + small font back plate = potential mule (era mismatch)

**Box height thresholds** (based on testing):
- Large font (modern): ~15px+ box height
- Small font (old/mule): ~13px box height

### New CSV/GUI Columns
- **Series**: Series year with suffix (e.g., "2013", "2017A")
- **Front Plate**: Front plate number (e.g., "FW I 30", "G144")
- **Back Plate**: Back plate number (now reads both font sizes)
- **Mule?**: "Yes" if potential mule detected

### Files Modified
- `settings_manager.py`: Added `extract_plate_info` setting
- `gui/settings_dialog.py`: Added checkbox in Processing tab
- `process_production.py`: Added `_extract_plate_info()` method, updated BillPair dataclass
- `gui/processing_thread.py`: Plate extraction during processing
- `gui/monitor_thread.py`: Plate extraction during monitor mode
- `gui/results_list.py`: New columns, CSV fieldnames, backward compatibility
- `gui/main_window.py`: Pass setting to threads, updated CSV export

### Mule-Hunting Series
Prime series for finding mules (transition period):
- Series 1988A
- Series 1993
- Series 1995

## Plate Magnifier Popup (Added Feb 2025)

### Overview
Quick visual comparison of front and back plate regions at 200% zoom for mule font-size identification.

### Usage
- Press **'M'** while viewing a bill to show the popup
- Press **'M'** again or **Escape** to close
- Shows front plate (from front image) and back plate (from back image) side-by-side

### How It Works
1. Uses YOLO to detect `front_plate` (class 4) and `back_plate` (class 0) regions
2. Applies cached alignment (rotation/flip) from current result
3. Crops regions with 10px padding
4. Scales to 200% zoom for easier font comparison
5. Displays in frameless popup dialog

### Files
- `gui/plate_magnifier_dialog.py`: PlateMagnifierDialog class (new)
- `gui/preview_panel.py`: `_extract_plate_regions()`, `show_plate_magnifier()` methods
- `gui/main_window.py`: 'M' keyboard shortcut

### Error Handling
| Scenario | Behavior |
|----------|----------|
| No bill selected | Silent return |
| No YOLO model | Silent return |
| Front plate not detected | Shows "Not detected" placeholder |
| Back plate not detected | Shows "Not detected" placeholder |
| No back image | Shows "Not detected" for back plate |

## Session Recovery & Autosave (Added Feb 2025)

### Overview
Automatic session state preservation to protect against crashes and power loss. If the app closes unexpectedly, progress can be restored on next startup.

### How It Works
- **Periodic autosave**: Every 30 seconds (configurable), session state is saved to `.session_recovery.json`
- **Atomic writes**: Uses temp file + rename for data integrity
- **Dirty flag tracking**: Only saves when data actually changes
- **Auto-clear on archive**: Recovery file deleted after successful archive

### Settings
- **Settings → Processing → "Enable autosave"**: Toggle on/off (default: on)
- **Settings → Processing → "Save interval"**: 10-300 seconds (default: 30)

### Recovery Flow
1. On startup, checks for `.session_recovery.json`
2. If found, shows Recovery Dialog with session info
3. User chooses "Restore Session" or "Discard & Start Fresh"
4. If restored, YOLO processor loads automatically for cropping/alignment

### Recovery File Contents
```json
{
  "version": 1,
  "timestamp": "2026-02-02T14:30:00",
  "input_directory": "/path/to/input",
  "results": [...],
  "processing_complete": true,
  "total_processed": 1000,
  "last_selected_index": 800
}
```

### Key Features
- **Lazy processor loading**: YOLO loads on-demand for restored sessions
- **Alignment data preserved**: `front_align_angle` and `front_align_flipped` saved/restored
- **Boolean normalization**: Handles string "True"/"False" conversion on load
- **Config loading**: Lazy processor loads `config.yaml` for crop settings

### Files
- `session_recovery.py`: SessionRecoveryManager class
- `gui/recovery_dialog.py`: Recovery prompt dialog
- `settings_manager.py`: AutosaveSettings dataclass
- `gui/settings_dialog.py`: Autosave UI controls
- `gui/main_window.py`: Timer, recovery check, lazy processor creation

### Bill Labels
When generating crops, `bill_labels.txt` is created/appended with:
```
Serial: A12345678B
Pattern: RADAR
Series: 2013
------------------------------

Serial: A12345678B
Pattern: RADAR
Series: 2013
Catalog: A1  Pos: 5
==============================
```
Each bill gets two labels - one without catalog (for binder) and one with catalog + position (to store with bill).

## Pattern Settings Migration (Added Feb 2025)

### Problem Solved
Previously, pattern customizations (like GAS_PUMP threshold) were stored in `user_patterns.yaml`, which got overwritten by `update.bat` when copying `*.yaml` files. This caused:
- Gas pump threshold resetting from user value (e.g., 3.6) to default (3.5)
- Custom pattern overrides being lost
- Disabled/enabled pattern preferences resetting

### Solution
All user-customizable pattern data is now stored in `user_settings.yaml`, which is gitignored and preserved across updates.

### What's Stored in user_settings.yaml
```yaml
pattern_overrides:
  GAS_PUMP:
    baseline_variance_min: 3.6
  OTHER_PATTERN:
    some_rule: value
pattern_states:
  PATTERN_A: false  # disabled
  PATTERN_B: true   # enabled
custom_patterns:
  MY_BIRTHDAY:
    description: "My special date"
    tier: 10
    rules:
      contains: "0704"
```

### Migration
On first run after update, `SettingsManager` automatically migrates data from old `user_patterns.yaml`:
- `pattern_overrides` → `settings.pattern_overrides`
- `disabled_patterns` → `settings.pattern_states[name] = False`
- `enabled_patterns` → `settings.pattern_states[name] = True`
- `custom_patterns` → `settings.custom_patterns`

Only migrates values not already present (won't overwrite).

### API Changes

**SettingsManager new methods:**
```python
# Pattern rule overrides (any pattern, any rule)
settings.get_pattern_override('GAS_PUMP', 'baseline_variance_min', default=3.5)
settings.set_pattern_override('GAS_PUMP', 'baseline_variance_min', 3.6)

# Convenience methods for GAS_PUMP threshold
settings.get_gas_pump_threshold(default=3.5)
settings.set_gas_pump_threshold(3.6)

# Custom YAML patterns
settings.get_custom_pattern('MY_PATTERN')
settings.set_custom_pattern('MY_PATTERN', defn)
settings.remove_custom_pattern('MY_PATTERN')
```

**PatternEngine changes:**
- Now accepts optional `settings` parameter: `PatternEngine(settings=my_settings)`
- Falls back to `get_settings()` singleton if not provided
- `get_gas_pump_threshold()` / `set_gas_pump_threshold()` delegate to SettingsManager
- `save_config()` syncs to SettingsManager instead of `user_patterns.yaml`

### Files Modified
- `settings_manager.py`: Added `pattern_overrides`, `custom_patterns`, migration logic
- `pattern_engine_v2.py`: Integrated with SettingsManager
- `gui/pattern_dialog.py`: Threshold editor uses SettingsManager
- `update.bat`: Selectively copies YAML files (skips `user_patterns.yaml`)
- `.gitignore`: Added `user_patterns.yaml`

### Verification
```bash
# Check threshold persists after update
python -c "
from settings_manager import get_settings
s = get_settings()
print('Gas pump threshold:', s.get_gas_pump_threshold())
print('Pattern overrides:', s.pattern_overrides)
"
```
