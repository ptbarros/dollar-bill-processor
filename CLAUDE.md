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
    -- ctx.metadata: {baseline_variance, seal_x, seal_y, seal_containment, series_year, front_plate, back_plate}

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

## Seal Shift Detection

Detects overprint misalignment by comparing treasury seal to "ONE" text underneath.

**YOLO v9 Classes:** `ONE_hashed`(0), `back_plate`(1), `bill_back`(2), `bill_front`(3), `denomination`(4), `front_plate`(5), `seal_f`(6), `seal_t`(7), `serial_number`(8), `series_year`(9), `star_symbol`(10)

**Metrics:**
- `seal_x/seal_y`: Offset as % of ONE_hashed dimensions
- `seal_containment`: % of seal inside ONE bbox (100% = normal, <97% = shifted)

**Pattern:** `SEAL_SHIFT` triggers when containment < 97%

## Plate Info & Mule Detection

Settings → Processing → "Extract plate and series info"
- Extracts series_year, front_plate, back_plate
- Mule detection: 1988+ series with back_plate height ≤14px
- Press **M** for plate magnifier popup

## GUI Features

**Keyboard:**
| Key | Action |
|-----|--------|
| Space | Queue bill for crop (toggle ✓) |
| C | Batch crop all queued |
| M | Plate magnifier |

**Context Menu:** "Set Pattern..." (override label), "Set Note..." (add comment)

**Layouts:** View > Layout menu (Classic, Wide Preview, Details Right)

**Zoom:** Fit/+/- buttons, Ctrl+scroll, middle-mouse drag

## Settings

Stored in `user_settings.yaml` (gitignored):
- UI: theme, layout_mode, font_size, default_fancy_color
- Pattern states/colors, library states/colors
- AI: provider, API keys, models
- Processing, export, monitor settings

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
