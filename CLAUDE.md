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
  - Lua patterns take precedence over YAML patterns with same name
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
- **highlights**: Color individual digit positions
- **connectors**: Lines between digit pairs (styles: arc, line, bracket, arrow, dashed)
- **group_boxes**: Single box spanning multiple digits (e.g., for MINI_REPEATER's "680" groups)

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
- YAML patterns remain for metadata (odds, prices) - Lua overrides detection logic
