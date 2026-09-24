# Dollar Detective - Project Memory

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
    -- ctx.data / ctx.data_by_key: loaded from DataFile (if specified).
    --   READ-ONLY: this is a shared cached Lua table (converted once per sandbox,
    --   not a private per-run copy) -- never assign into it, or you poison every
    --   later match. See pattern_sandbox._convert_to_lua_cached.
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
- **highlights**: Individual digit boxes. Optional `style` per highlight: `"box"` (default), `"x"` (an X mark, no box), or `"boxed_x"` (box + X) — use `style="x"` with `color="gray"`/`"charcoal"` to mark an EXCLUDED/leftover digit clearly (an X can't be mistaken for a match the way a muted box can). e.g. `{positions={5}, color="charcoal", style="x"}`
- **connectors**: Lines between digits (styles: arc, line, bracket, arrow, dashed)
- **group_boxes**: Box spanning multiple digits (preferred for groups)
- **Colors**: pick from the strong palette `blue, orange, magenta, red, purple, hotpink` (+ `black` for max contrast, `gray` for muted/prefix). Colors are for VISIBILITY, not to signal pattern type. Use *different* names to distinguish sub-groups within one pattern; the overlay auto-remaps requested names onto a contrast-tested rotation (first-seen → blue, then orange, magenta, …) so drawn colors are always strong and mutually distinct. Legacy/weak names (`cyan, teal, lime, green, yellow, gold, coral, salmon, white`) still work but are aliased to the nearest strong color. Palette derived by contrast-testing vs the measured bill background — see `serial_overlay.PATTERN_COLORS`.

### Pattern Dialog Features
- **Pattern Wizard**: GUI-based recipe creation (Ladder, Binary, Pairs, Palindrome, etc.)
- **AI Generate**: Natural language to Lua via Anthropic/OpenAI APIs
- **Test Tab**: Quick test, batch test cases, debug logging with `log()` function
- **Copy for AI**: Exports API docs + template for external AI tools
- **Export/Import Selection**: shares the enabled/disabled on-off list (JSON) — NOT the pattern definitions; only works if the recipient already has those patterns.
- **Export/Import Bundle**: shares the actual patterns as a single `.ddpat` file (a zip). Export packs the selected pattern(s) — or all user patterns if none selected — plus any `DataFile` CSV/JSON they use, plus each pattern's custom display-**label** override (manifest `pattern_labels`, so relabeling travels with the patterns). Import copies patterns into a **named add-on library folder** under the user-data patterns tree (name from the manifest `library` field, else the bundle filename), so it shows as its own group (e.g. "Green Guide") and can be removed as one — not merged into the flat `user` folder. Data files written as siblings; `DataFile:` header normalized to the basename so it resolves; bundled labels applied via `settings.set_pattern_label`; name collisions handled (skip/overwrite); then `engine.reload()`. The engine scans `user_data_dir/patterns/*` subdirs as libraries (`user_libraries_root`), and **Remove Library…** deletes a non-bundled add-on library (`engine.removable_libraries`/`remove_library`). Logic in `pattern_bundle.py`; wired via `_export_bundle`/`_import_bundle`/`_remove_library` in `gui/pattern_dialog.py`.- **Label overrides — bulk + backup**: Pattern Manager "Labels:" row has **Strip "CS-"** (bulk-remove the leading `CS-` from every effective label → per-pattern overrides), **Back Up…**/**Restore…** (JSON of `settings.pattern_labels`, format `dollar-detective-pattern-labels`; restore offers Merge/Replace). Overrides live in `user_settings.yaml` keyed by internal pattern name.

### Key Files
| File | Purpose |
|------|---------|
| `pattern_engine_v3.py` | Lua pattern engine |
| `pattern_sandbox.py` | Secure Lua execution |
| `gui/pattern_dialog.py` | Pattern Manager + CustomPatternDialog |
| `pattern_bundle.py` | Pattern bundle (.ddpat) export/import (patterns + data files + label overrides) |
| `gui/label_render.py` | Label drawing + profile template (LabelTemplate/LabelField: size, base font, fields, per-field fonts) — single source for preview + PDF/Word export (wrap, overflow) |
| `gui/label_preview_dialog.py` | Label Preview & Print tool (Tools → Label Preview, Ctrl+Shift+L); per-bill edits + profile picker |
| `gui/label_fields_dialog.py` | Label PROFILE editor (Edit… in Label Preview): label size, base font, fields, captions, per-field fonts |
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
- Mule detection (EXPERIMENTAL hint, not a verdict): flags a pre-1960s series (1928 ≤ year < 1960) whose back-plate YOLO box height ≤14px (a "small font" proxy). Shown as "Check" in the "Mule? (exp)" column. **This is a weak heuristic** — a true mule is a micro/macro plate-number *font-size mismatch* between face and back on small-size notes (~1930s–50s), denomination/plate-number specific and not reliably readable from a scan. The magnifier is the real tool.
- Press **M** for plate magnifier popup (front/back plates side-by-side to eyeball the micro/macro font — the actual mule check)

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

## Downstream consumer: DollarDetective Web

The web spinoff (`~/projects/dollardetective-web`, live at dollardetective.tarso.net)
**vendors these files from this repo** via its `scripts/sync-engine.sh` — it reuses the
desktop engine server-side:

- `pattern_engine_v3.py`, `pattern_sandbox.py`, `resource_path.py`
- `patterns/{lib,core}/*.lua`
- `patterns/user/1959.lua` (Paul's own user pattern)

**Deleting or renaming any of these breaks the web app's next sync**, and an unguarded
read of a vanished file can stop it at startup (this happened when the `Essentials`
library was removed, 2026-09-16). Before landing such a change, flag it to the web side;
if that isn't practical, the web app can absorb it by guarding the read. The web repo
does NOT auto-update — pattern changes need a re-vendor + rebuild there.

## Notes

- User patterns in `patterns/user/` are gitignored
- `engine.reload()` reloads all patterns
- Low run patterns: LOW_RUN_6M (Tier 5), LOW_RUN_12M (Tier 6). Data in `patterns/core/low_runs.csv`; see `patterns/core/LOW_RUNS.md` for how the data is derived from uspapermoney.io serial charts (96M block cap, facility slices) and `tools/parse_low_runs.py` to regenerate candidates from a saved chart.
- Debug logging: Use `log()` in Lua patterns during batch testing

## Pattern libraries

All shipped patterns now live in a single flat `patterns/core/` library (the old
`Nicks` and `The Green Guide` folders were flattened into it, then culled to the
current set). The Green Guide dependency has been removed entirely: no `BookRef:`
headers, no CS references in messages/descriptions/comments, and the app no longer
ships the Green Guide `.ddpat` bundle. Do not reintroduce CS numbering or book
nomenclature. (Four names are still under review with Ed — see project memory —
so leave True Binary Alternator / True Double Quad Binary / Binary Quads /
Rotator 018 alone until he decides.)
