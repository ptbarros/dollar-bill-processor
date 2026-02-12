# Dollar Bill Processor - Project Memory

## Overview
A GUI application for processing dollar bill images, detecting serial numbers via OCR, and classifying them against collectible "fancy serial number" patterns.

## Lua Pattern Plugin System

### Architecture
- **pattern_sandbox.py**: Secure Lua execution environment using `lupa` library
- **pattern_engine_v3.py**: Lua-only pattern engine (standalone, no YAML)
- **patterns/** directory structure:
  ```
  patterns/
  ├── core/              # Built-in Lua patterns (123 patterns)
  ├── Nicks/             # Nick's pattern library (62 patterns)
  ├── user/              # User-created patterns (gitignored)
  ├── <custom library>/  # Any folder becomes a library
  ├── lib/helpers.lua    # Shared utility functions
  └── data/              # Shared data files (CSV/JSON)
  ```

### Lua Script Structure
```lua
--[[
Pattern: PATTERN_NAME
DisplayName: Friendly Name With Spaces
Description: What it matches
Tier: 1-10
Examples: ["12345678"]  -- REQUIRED for random preview generator!
Odds: 1 in 10,000
Price: $20-$100
DataFile: optional_data.csv
--]]

function match(ctx)
    -- ctx.digits: "12345678" (8 numeric characters)
    -- ctx.full_serial: "A12345678B" (with prefix/suffix)
    -- ctx.digit_list: {1,2,3,4,5,6,7,8} as integers
    -- ctx.data: loaded from DataFile (if specified)
    -- ctx.data_by_key: dict keyed by first CSV column
    -- ctx.metadata: {baseline_variance, seal_x, seal_y, series_year, front_plate, back_plate}

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
- **highlights**: Color individual digit positions (each gets own box)
- **connectors**: Lines between digit pairs (styles: arc, line, bracket, arrow, dashed)
- **group_boxes**: Single box spanning multiple digits (preferred for multi-digit groups)
- **Color palette**: purple, blue, cyan, orange, coral, gold, salmon, magenta, yellow, lime, green, teal, red, gray

Best practices:
- Use `group_boxes` for bookends and repeating groups
- Use one arc per group, not per digit
- Color coding: lime (ascending), cyan (descending), yellow (peaks), orange (bookends)

### External Data Files
Lua patterns can declare CSV/JSON dependencies loaded into `ctx.data`:
```lua
--[[
DataFile: known_serials.csv
--]]
```
- Filename only → same directory as .lua file
- `data/file.csv` → `patterns/data/` directory
- CSV loaded as list + `ctx.data_by_key` (keyed by first column)
- Data cached at startup, reloaded on `engine.reload()`

### Pattern Manager GUI
- Library-based organization with collapsible sections
- Library checkbox enables/disables all patterns in that library
- Color customization: pattern color > library color > default fancy color
- View Script (core, read-only) / Edit Script (user, opens full editor dialog)
- Visual preview with random serial generation and live testing
- DisplayName auto-generated from pattern name if not specified
- Preview works even for disabled patterns (temporarily enables for display)

### AI-Assisted Pattern Scripting (CustomPatternDialog)

The New/Edit Pattern dialog includes features to streamline pattern creation with external AI tools:

**API Docs Tab:**
- "Copy API Docs" - Copy API reference to clipboard
- "Copy for AI" - Copy comprehensive prompt including:
  - Context preamble for AI
  - Full API documentation
  - Helper function reference (extracted from helpers.lua)
  - Fill-in-the-blank template with `[PATTERN_NAME]`, `[DESCRIPTION]`, etc.
  - Validation rules and common pitfalls

**Test Tab:**
- Quick Test: Single serial input with live preview
- Batch Test Cases: Two text areas for "Should Match" and "Should NOT Match" serials
- "Run All Tests" - Execute all test cases, show pass/fail with debug logs inline
- "Export for AI" - Copy test cases formatted for AI prompts
- "Copy for AI Debug" - Enabled when tests fail; copies script + ctx contents + debug logs

**Save Validation:**
- Warning dialog if Examples field is missing (required for random preview generator)

### Pattern Wizard (Recipe-Based Creation)

The "Pattern Wizard" tab provides GUI-based pattern creation for non-coders:

**Recipe Types:**
- **Ladder/Sequence**: Ascending/descending digit runs (min length 4-8)
- **Digit Set Restriction**: Match only specific digits (presets: Binary, Flipper, Evens, Odds)
- **Repeating Patterns**: Consecutive pairs, Repeater, Super Repeater, Alternator
- **Palindrome/Radar**: Exact or near-palindromes (1-2 mismatches)
- **Digit Sum**: Match by digit sum (exact value or range)
- **Bookends**: First N digits = last N digits

**Features:**
- Dynamic parameter widgets based on selected recipe
- Live preview with generated examples
- Collapsible "Generated Lua Code" section
- Automatic Examples generation for preview compatibility

**Files:** `gui/pattern_recipes.py` (recipe infrastructure)

### AI Generate (Integrated AI Pattern Creation)

The "AI Generate" tab provides natural language to Lua code generation:

**Workflow:**
1. Describe pattern in plain English
2. Optionally provide "Should match" / "Should NOT match" examples
3. Select provider from dropdown (Anthropic or OpenAI)
4. Click "Generate Pattern" - calls selected AI API
5. Review generated code in preview
6. Click "Use This Code" to copy to Lua Script tab (auto-prefills Pattern Info from header)
7. Test and save

**AI Prompt Features:**
- Comprehensive helper function documentation with examples
- Warnings for common gotchas (nil returns, pair spacing math, Lua built-in shadowing, etc.)
- Automatic Examples injection if AI omits them

**Configuration:** Settings → AI tab
- Provider selection (default provider for new sessions)
- Separate API keys for Anthropic and OpenAI (both can be configured)
- Model selection (editable dropdowns)
- Test Connection button

**In-Tab Provider Selection:**
- AI Generate tab has its own provider dropdown
- Only shows providers with configured API keys
- Switch providers without leaving the pattern dialog

**Files:** `gui/ai_pattern_generator.py` (API client and prompt building)

**Dependencies:** `pip install anthropic` and/or `pip install openai`

### Key Files
- `pattern_engine_v3.py`: Lua-only pattern engine
- `pattern_sandbox.py`: Secure Lua execution environment
- `gui/pattern_dialog.py`: Pattern Manager dialog (includes CustomPatternDialog with Wizard/AI tabs)
- `gui/pattern_recipes.py`: Recipe-based pattern creation (6 recipe types)
- `gui/ai_pattern_generator.py`: AI pattern generation (Anthropic/OpenAI)
- `gui/preview_panel.py`: Pattern visualization
- `gui/settings_dialog.py`: Settings dialog (includes AI configuration tab)
- `settings_manager.py`: User settings persistence (includes AISettings)
- `process_production.py`: Main processing pipeline

## Helper Functions (patterns/lib/helpers.lua)

### Core Analysis
- `count_digits(s)`, `find_runs(s)`, `unique_count(s)`, `digit_sum(s)`
- `most_common(s)`, `get_unique_digits(s)`

### Pattern Detection
- `is_ladder(s)`, `is_ascending(s)`, `is_descending(s)`
- `find_ladder_of_length(s, min_length)`, `find_longest_ladder(s)`
- `is_palindrome(s)`, `is_broken_palindrome(s, max_mismatches)`
- `is_repeater(s)`, `is_super_repeater(s)`, `is_alternating(s)`
- `has_n_consecutive(s, n)`

### Flipper Functions
- `all_flip_valid(s)`, `flip_string(s)`

### Pair/Group Detection
- `find_pairs(s)`, `find_consecutive_pairs(s)`, `count_pairs(s)`
- `has_four_consecutive_pairs(s)`, `has_three_consecutive_pairs_start(s)`
- `find_triples(s)`, `find_quads(s)`

### String Utilities
- `starts_with(s, prefix)`, `ends_with(s, suffix)`, `contains(s, substr)`
- `only_digits(s, allowed)`, `is_bookended(s, n)`

### Visualization Helpers
- `highlight(positions, color, label)`, `highlight_range(start, stop, color, label)`
- `connector(from, to, color, style)`
- `find_digit_positions(s, digit)`, `find_matching_positions(s, digit_set)`

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

# Verify example serials match their patterns
python -c "
from pattern_engine_v3 import PatternEngineV3
engine = PatternEngineV3()
for name, info in engine.lua_patterns.items():
    if info.examples:
        for ex in info.examples:
            if len(ex) == 8 and ex.isdigit():
                matches = engine.classify_simple(f'A{ex}B')
                if name not in matches:
                    print(f'{name}: example {ex} matches {matches} instead!')
"
```

## Plate Info Extraction & Mule Detection

Optional feature (Settings → Processing → "Extract plate and series info"):
- Extracts series_year, front_plate, back_plate using YOLO + OCR
- Detects potential mule notes (1988+ series with small-font back plate)

**Front plate format:** `[FW] [A-J] [digits]`
- Uses contour-based detection (check letter is taller than other characters)
- Fallback OCR with post-processing corrections

**Mule detection:** Series 1988+ with back_plate box height ≤14px = potential mule

**Keyboard:** Press 'M' to show plate magnifier popup (200% zoom comparison)

## Overprint Shift Detection (Seal Shift)

Detects overprint misalignment by comparing the treasury seal to the spelled-out "ONE" text underneath it.

### YOLO Model v9 Classes
The v9 model adds `ONE_hashed` class, causing all class IDs to shift +1:
```
ID  Name
--  ----
 0  ONE_hashed      ← NEW (spelled-out "ONE" under treasury seal)
 1  back_plate
 2  bill_back
 3  bill_front
 4  denomination
 5  front_plate
 6  seal_f
 7  seal_t
 8  serial_number
 9  series_year
10  star_symbol
```

### Algorithm (Seal vs ONE_hashed)
The treasury seal is physically printed on top of the "ONE" text. When the overprint plate shifts, the seal drifts outside the ONE bounding box.

1. Run YOLO inference at conf=0.1 to detect `seal_t` and `ONE_hashed`
2. Compute center-to-center offset as % of ONE_hashed dimensions:
   ```python
   dx_pct = (seal_cx - one_cx) / one_w * 100
   dy_pct = (seal_cy - one_cy) / one_h * 100
   ```
3. Compute containment (% of seal area inside ONE bbox):
   ```python
   containment = intersection_area / seal_area * 100
   ```

**Why seal vs ONE?**
- Zero extra YOLO calls - both classes from same inference pass
- Pure box arithmetic (microseconds)
- Per-image measurement - no cross-image baseline needed
- Direct geometric relationship - seal is physically on top of "ONE"

### Metrics (Standard Coordinate System)
Uses standard X/Y coordinates: +x = right, -x = left, +y = up, -y = down

- `seal_x`: X offset as % of ONE_hashed width (+x = right, -x = left)
- `seal_y`: Y offset as % of ONE_hashed height (+y = up, -y = down)
- `seal_containment`: % of seal area inside ONE bbox (100% = normal)

**Normal range:** dy from -1.0% to +1.6%, containment 100%
**Shifted bills:** dy -6.87% (down) or +8.51% (up), containment ~94%

### Pattern
- `SEAL_SHIFT`: Triggers when **containment < 97%** (single threshold)
  - Message includes direction from Y value: "Seal shift (6.6% down, 95% contained)"

**CSV output:** `seal_x`, `seal_y`, `seal_containment` columns

**GUI display:**
- "Shift X%" - horizontal offset (%)
- "Shift Y%" - vertical offset (%, +up/-down)
- "Seal %" - containment (100% = normal, <97% = shifted)

**Lua metadata:** Access via `ctx.metadata.seal_y` and `ctx.metadata.seal_containment`

**Debug tool:**
```bash
./venv/bin/python tools/debug_seal_detection.py ~/Pictures/Dollar/seal_test/
```

## Session Recovery & Autosave

- Periodic autosave to `.session_recovery.json` (configurable interval)
- On startup, prompts to restore or discard recovered session
- Tracks: viewed, cropped, sent_for_review, checked status per bill
- Status changes mark session dirty for autosave

**Keyboard shortcuts:**
| Key | Action |
|-----|--------|
| Space | Queue bill for crop (toggle ✓ status) |
| C | Batch crop all queued bills |
| M | Plate magnifier popup |

**Context menu (right-click):**
- "Set Pattern..." - Choose which pattern appears in `bill_labels.txt` (stores override, no immediate crop)
- "Set Note..." - Add a user note to the bill (appears in labels file)

**Crop Workflow:**
1. Review bills - press **Space** to queue keepers (shows ✓)
2. Right-click → "Set Pattern..." to override label pattern (if multi-pattern bill)
3. Right-click → "Set Note..." to add comment (e.g., "Birthday serial")
4. Press **C** to batch crop all queued bills
5. After crop: ✓ clears, C status appears

## Layout Modes

Switchable panel layouts via **View > Layout** menu. Preference saved to `user_settings.yaml`.

**Available layouts:**

| Layout | Description |
|--------|-------------|
| **Classic** (default) | Results list on left, Preview + Serial + Details stacked on right |
| **Wide Preview** | Preview + Serial on top (full width), Results list below |
| **Details Right** | Preview + Serial on top, Results + Details side-by-side below |

**Details Right layout:**
```
+-------------------------------------------+
|         ProcessingPanel                   |
+-------------------------------------------+
|          Bill Preview                     |
|          + Serial Region                  |
+-------------------+-----------------------+
|   Results List    |    Bill Details       |
+-------------------+-----------------------+
```
The divider between Results and Details is adjustable.

**Files:**
- `gui/layout_manager.py`: LayoutManager class, layout constants
- `settings_manager.py`: `UISettings.layout_mode` field

## User Settings Persistence

All user customizations stored in `user_settings.yaml` (gitignored):
```yaml
ui:
  layout_mode: classic  # or "wide_preview", "details_right"
pattern_overrides:
  GAS_PUMP:
    baseline_variance_min: 3.6
pattern_states:
  PATTERN_A: false
pattern_colors:
  STAR: '#c061cb'
library_states:
  core: true
  Nicks: true
library_colors:
  Nicks: '#ff6600'
ai:
  provider: anthropic  # or "openai"
  anthropic_api_key: sk-ant-...
  openai_api_key: sk-...
  anthropic_model: claude-sonnet-4-20250514
  openai_model: gpt-4o
```

Access via SettingsManager:
```python
settings.get_pattern_override('GAS_PUMP', 'baseline_variance_min', default=3.5)
settings.set_pattern_override('GAS_PUMP', 'baseline_variance_min', 3.6)
settings.get_gas_pump_threshold()   # convenience method (3.5 default)
```

## Synthetic Test Bill Generator

Generate test bills with specific serials for pattern regression testing:
```bash
python tools/create_test_bill.py --serial "A12344321B" --results archive/*/results.csv --output-dir test_bills/
python tools/create_test_bill.py --pattern RADAR --count 5 --results archive/*/results.csv --output-dir test_bills/
python tools/create_test_bill.py --all-patterns --results archive/*/results.csv --output-dir test_bills/
```

Composites character glyphs from real scanned bills. Skipped patterns: GAS_PUMP, STAR, LOW_RUN_*, KNOWN_SERIALS, SEAL_SHIFT, HIGH_SEAL, LOW_SEAL (require metadata or special images).

## Notes

- User patterns in `patterns/user/` are gitignored
- Helpers loaded automatically into Lua sandbox
- `engine.reload()` reloads all patterns
- `get_pattern_info()` returns both `price` and `price_range` for backward compatibility
- Low run patterns split into LOW_RUN_6M (Tier 5) and LOW_RUN_12M (Tier 6)
- `.gitignore` exception for `patterns/core/low_runs.csv`

## TODO: Lua Pattern Debugging / Diagnostics

Users have no visibility into why patterns silently fail. Possible approaches:
1. Debug mode in Live Preview showing `ctx` contents
2. Pattern load diagnostics (parsed header, data file status)
3. ~~`log()` function in Lua sandbox surfacing to GUI~~ (DONE - see Debug Logging section below)
4. Header validation warnings (duplicate blocks, missing DataFile)
5. ~~Test harness with step-by-step trace output~~ (partially addressed via batch testing in CustomPatternDialog)

## Debug Logging for Pattern Scripts

Lua patterns can use `log()` to output debug information during batch testing:

```lua
function match(ctx)
    log("digits:", ctx.digits)
    log("digit_list:", ctx.digit_list)

    local count = unique_count(ctx.digits)
    log("unique count:", count)

    if count <= 2 then
        log("matched!")
        return {matched = true, message = "Binary"}
    end

    log("no match, count was", count)
    return {matched = false}
end
```

**Features:**
- `log(value1, value2, ...)` - accepts multiple values, space-separated
- Tables are serialized as `{key=value, ...}` format
- Logs appear inline with batch test results in the Test tab
- Logs are included in "Copy for AI Debug" output
- Zero overhead in production (debug mode only enabled during testing)

## TODO: Batch Test Cases Layout Compression Issue

In CustomPatternDialog's Test tab, the "Batch Test Cases" section gets squashed/compressed when the window is small, while other sections (Quick Test, Visual Preview, Test Results) maintain their size. The buttons and input fields overlap.

**Location:** `gui/pattern_dialog.py`, `_create_test_tab()` method, around line 2283

**Symptoms:**
- "Should NOT Match:" label overlaps the input field above it
- "Run All Tests" / "Export for AI" buttons get squashed into the input field above
- Only affects Batch Test Cases section, not the other three sections

**Attempted fixes that did NOT work:**
- `setMinimumSize(700, 800)` on the dialog
- `resize(700, 800)` after `_setup_ui()`
- `QTimer.singleShot(0, lambda: self.resize(700, 800))` for delayed resize
- `setMinimumHeight(25)` on QLineEdit inputs
- `setMinimumHeight(28)` on QPushButton buttons
- `addSpacing(8)` between elements
- `setMinimumHeight(140)` on the batch_group QGroupBox
- `setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)` on batch_group
- Removing `layout.addStretch()` at end of test tab

**Current state:** Usable after manual window resize. Low priority.
