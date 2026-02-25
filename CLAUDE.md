# Dollar Bill Processor - Project Memory

## Overview
GUI application for processing dollar bill images, detecting serial numbers via OCR, and classifying them against collectible "fancy serial number" patterns.

## Lua Pattern Plugin System

### Architecture
- **pattern_engine_v3.py**: Lua-only pattern engine
- **pattern_sandbox.py**: Secure Lua execution environment (`lupa` library)
- **patterns/**: `core/` (built-in), `Nicks/`, `user/` (gitignored), `lib/helpers.lua`, `data/`

### Lua Script Structure
```lua
--[[
Pattern: PATTERN_NAME
DisplayName: Friendly Name
Description: What it matches
Tier: 1-10
Examples: ["12345678"]  -- REQUIRED for preview generator
Odds: 1 in 10,000
Price: $20-$100
DataFile: optional_data.csv
--]]

function match(ctx)
    -- ctx.digits: "12345678" (8 numeric characters)
    -- ctx.full_serial: "A12345678B" (with prefix/suffix)
    -- ctx.digit_list: {1,2,3,4,5,6,7,8} as integers
    -- ctx.data: loaded from DataFile (if specified)
    -- ctx.metadata: {baseline_variance, gas_pump_threshold, seal_x, seal_y, seal_containment, series_year, front_plate, back_plate}

    return {
        matched = true,
        highlights = {{positions = {0, 1}, color = "orange"}},
        connectors = {{from = 0, to = 7, color = "orange", style = "arc"}},
        group_boxes = {{from = 0, to = 2, color = "gold", thickness = 3}},
        message = "Description of match"
    }
end
```

### Visualization
- **highlights**: Individual digit boxes
- **connectors**: Lines between digits (styles: arc, line, bracket, arrow, dashed)
- **group_boxes**: Box spanning multiple digits (preferred for groups)
- **Colors**: purple, blue, cyan, orange, coral, gold, salmon, magenta, yellow, lime, green, teal, red, gray

### Pattern Dialog Features
- **Pattern Wizard**: GUI-based recipe creation (Ladder, Binary, Pairs, Palindrome, etc.)
- **AI Generate**: Natural language to Lua via Anthropic/OpenAI APIs
- **Test Tab**: Quick test, batch test cases, debug logging with `log()` function
- **Copy for AI**: Exports API docs + template for external AI tools

### Key Files
| File | Purpose |
|------|---------|
| `pattern_engine_v3.py` | Lua pattern engine |
| `pattern_sandbox.py` | Secure Lua execution |
| `gui/pattern_dialog.py` | Pattern Manager + CustomPatternDialog |
| `gui/pattern_recipes.py` | Recipe-based creation |
| `gui/ai_pattern_generator.py` | AI pattern generation |
| `settings_manager.py` | User settings persistence |
| `process_production.py` | Main processing pipeline |

## Helper Functions (patterns/lib/helpers.lua)

**Analysis:** `count_digits`, `find_runs`, `unique_count`, `digit_sum`, `most_common`, `get_unique_digits`

**Pattern Detection:** `is_ladder`, `is_ascending`, `is_descending`, `find_ladder_of_length`, `is_palindrome`, `is_broken_palindrome`, `is_repeater`, `is_super_repeater`, `is_alternating`, `has_n_consecutive`

**Pairs/Groups:** `find_pairs`, `find_consecutive_pairs`, `count_pairs`, `find_triples`, `find_quads`

**String:** `starts_with`, `ends_with`, `contains`, `only_digits`, `is_bookended`

**Visualization:** `highlight`, `highlight_range`, `connector`, `find_digit_positions`

## Gas Pump Detection

Detects vertically misaligned digits (mechanical counter rollover during printing).

**Single method:** `analyze_gas_pump_digits()` in `process_production.py` — used by both processing and the overlay. Processing analyzes ALL serial boxes on the bill front and takes the max deviation to match what the overlay displays.

**Threshold:** Controlled by the Gas Pump slider in the overlay panel. Stored in `user_settings.yaml` as `pattern_overrides.GAS_PUMP.baseline_variance_min`. Passed to the Lua pattern via `ctx.metadata.gas_pump_threshold` (default 3.5px). Changing the slider + Re-classify updates results.

**GPT column:** Shows `max_deviation` (pixels) — the largest vertical offset of any digit from the median baseline, across all serial regions on the bill.

## Seal Shift Detection

Detects overprint misalignment by comparing treasury seal to "ONE" text underneath.

**Metrics:**
- `seal_x/seal_y`: Offset as % of ONE_hashed dimensions
- `seal_containment`: % of seal inside ONE bbox (100% = normal, <97% = shifted)

**Pattern:** `SEAL_SHIFT` triggers when containment < 97%

## Plate Info & Mule Detection

Settings → Processing → "Extract plate and series info"
- Extracts series_year, front_plate, back_plate
- Mule detection: 1988+ series with back_plate height ≤14px
- Press **M** for plate magnifier popup

## Performance Optimization

### Organize Folder (Orange Button)
Pre-processes folder for faster subsequent processing:
- Classifies front/back, fixes orientation, corrects skew
- Renames to `Dollar_NNN.jpg` format (odd=front, even=back)
- After organizing: verify is skipped, YOLO alignment is skipped

### YOLO Caching
- `verify_and_swap_pairs()` caches detections in `BillPair.front_cache/back_cache`
- `classify_and_cache_image()` extracts all detection data in one YOLO call (conf=0.1)
- Cached data reused by `align_image()` and `extract_serial()`

### Format Detection
- `dollar_sequential`: Pre-organized Dollar_NNN.jpg files (fastest path)
- `suffix`: Files with `_b` suffix (e.g., 0001.jpg + 0001_b.jpg)
- `sequential`: Alternating numbered pairs

## Testing

```bash
# Test pattern engine
python pattern_engine_v3.py

# Test specific serial
python -c "
from pattern_engine_v3 import PatternEngineV3
engine = PatternEngineV3()
print(engine.classify_simple('A12344321B'))
"
```

## Notes

- User patterns in `patterns/user/` are gitignored
- `engine.reload()` reloads all patterns
- Low run patterns: LOW_RUN_6M (Tier 5), LOW_RUN_12M (Tier 6)
- Debug logging: Use `log()` in Lua patterns during batch testing

## Green Guide Pattern Library

**130 implemented patterns** in `patterns/The Green Guide/`. Full status in `patterns/The Green Guide/TRACKING.md`.

### Book sources
- `/tmp/tggfsn.txt` — OCR scan of the Green Guide book. @CS~NNN tags identify pattern numbers.
- `~/projects/tggfsn.ods` — Spreadsheet of all book patterns with accurate CS#, page numbers, chapter. **Use this as the authoritative CS# reference** — it was verified against the book appendix and corrected many wrong CS# assignments that were in previous sessions.

### ODS column layout (tggfsn.ods, col 0-based)
| Col | Header | Notes |
|-----|--------|-------|
| 0 | Page # | |
| 1 | Chapter | |
| 2 | Original order | |
| 3 | Skip | x = do not implement |
| 4 | Pattern Created | x = implemented as Lua |
| 5 | CS-# | authoritative CS number |
| 6 | Name | display name matching book |
| 7 | Examples | positional variant serials from book (M xx M format) |
| 8 | Description | book prose definition (manually verified) |

### CS# verified state (2026-02-24)
All 130 implemented pattern files have correct `BookRef:` fields verified against the spreadsheet. `tools/verify_patterns.py` passes **130/130** (once ODS is updated) with zero failures. Original 104 patterns: ODS Description (col 8) and Examples (col 7) fields manually verified. Batch 7 (19 patterns): CS# verified from ODS. Batch 8 (7 counting patterns): CS-830 through CS-890. Key corrections made:
- CS-100 = CS-Triple, CS-110 = CS-3OAK (not swapped)
- CS-190 = CS-4OAK, CS-200 = CS-Quad, CS-210 = CS-Random 4OAK
- CS-1060 = CS-Trinary Flipper, CS-1070 = CS-Quad Flipper
- CS-1260 = CS-Super Radar, CS-1370 = CS-Mini 3 Radar (not CS-1340)
- CS-1860 = CS-Stand Alone Mini Ladder (not CS-1880)
- CS-1340 = CS-Shotgun Radar (not yet implemented)
- CS-2280 = CS-Zip Codes, CS-2290 = CS-Prime Numbers

### Naming conventions (IMPORTANT — must match book exactly)
- **"OAK" = Of A Kind**: use `3OAK`, `4OAK`, `5OAK`, `6OAK`, `7OAK` — NOT `30AK`, `40AK` etc.
- **"CS-Random XXX"** prefix — NOT "CS-XXX (Random)" suffix
- **No invented qualifiers**: don't add "(Scattered)", "(Grouped)", "(CS-80AK)" etc. unless the book uses that exact wording
- Pattern family example: "CS-Quad Pairs" (grouped AABBCCDD), "CS-Random Quad Pairs" (scattered)
- DisplayName audit is **complete** — all original 104 files verified; batch 7 patterns written fresh to match book names

### Pending work
See `patterns/The Green Guide/TRACKING.md` Todo section for the full list.
