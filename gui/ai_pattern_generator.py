"""
AI Pattern Generator - Generate Lua patterns using AI APIs.

Supports both Anthropic (Claude) and OpenAI (GPT) APIs.
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class AIGenerationResult:
    """Result of an AI pattern generation request."""
    success: bool
    lua_code: str = ""
    error: str = ""
    raw_response: str = ""


class AIPatternGenerator:
    """Generate Lua pattern scripts using AI APIs."""

    def __init__(self, provider: str, api_key: str, model: str):
        """
        Initialize the generator.

        Args:
            provider: "anthropic" or "openai"
            api_key: API key for the provider
            model: Model name (e.g., "claude-sonnet-4-20250514" or "gpt-4o")
        """
        self.provider = provider
        self.api_key = api_key
        self.model = model

    def generate(self, description: str,
                 should_match: list[str] = None,
                 should_not_match: list[str] = None,
                 pattern_name: str = "",
                 progress_callback: Callable[[str], None] = None) -> AIGenerationResult:
        """
        Generate a Lua pattern from a natural language description.

        Args:
            description: Natural language description of the pattern
            should_match: Optional list of example serials that should match
            should_not_match: Optional list of example serials that should NOT match
            pattern_name: Optional suggested pattern name
            progress_callback: Optional callback for progress updates

        Returns:
            AIGenerationResult with the generated Lua code or error
        """
        if not self.api_key:
            return AIGenerationResult(success=False, error="API key not configured")

        if not description.strip():
            return AIGenerationResult(success=False, error="Please provide a pattern description")

        # Build the prompt
        prompt = self._build_prompt(description, should_match, should_not_match, pattern_name)

        # Call the appropriate API
        if self.provider == "anthropic":
            result = self._call_anthropic(prompt, progress_callback)
        elif self.provider == "openai":
            result = self._call_openai(prompt, progress_callback)
        else:
            return AIGenerationResult(success=False, error=f"Unknown provider: {self.provider}")

        # Post-process: inject Examples if missing and user provided some
        if result.success and result.lua_code:
            result.lua_code = self._ensure_examples(result.lua_code, should_match)

        return result

    def _build_prompt(self, description: str,
                      should_match: list[str] = None,
                      should_not_match: list[str] = None,
                      pattern_name: str = "") -> str:
        """Build the full prompt for the AI."""

        # Build examples section
        examples_section = ""
        if should_match:
            examples_section += f"\n**Should Match (examples):**\n"
            for ex in should_match[:10]:  # Limit to 10
                examples_section += f"- {ex}\n"

        if should_not_match:
            examples_section += f"\n**Should NOT Match (examples):**\n"
            for ex in should_not_match[:10]:  # Limit to 10
                examples_section += f"- {ex}\n"

        # Build suggested examples for the header
        suggested_examples = ""
        if should_match:
            import json
            suggested_examples = f'\nUse these as your Examples: {json.dumps(should_match[:5])}'

        prompt = f'''You are an expert Lua programmer helping write patterns for dollar bill serial number classification. These patterns analyze 8-digit serial numbers and return match results with optional visual highlighting.

## API Documentation

### Script Header (REQUIRED FORMAT)
```lua
--[[
Pattern: PATTERN_NAME
DisplayName: Friendly Name With Spaces
Description: What this pattern matches
Tier: 1-10 (1=rarest, 10=common)
Examples: ["12345678", "87654321"]
--]]
```

⚠️ **CRITICAL: The Examples field is MANDATORY.** You MUST include an Examples line with 3-5 example 8-digit serials that match the pattern. The application will not work without it.{suggested_examples}

### Input Context
The `ctx` table is available in every pattern script:
- `ctx.digits`: "12345678" (8 numeric characters)
- `ctx.full_serial`: "A12345678B" (with prefix/suffix letters)
- `ctx.digit_list`: {{1,2,3,4,5,6,7,8}} as integer array (1-indexed in Lua)

### Return Value
The match function must return a table with:
- `matched`: boolean (required - true if pattern matches)
- `highlights`: list of {{positions = {{0, 7}}, color = "orange"}}
- `connectors`: list of {{from = 0, to = 7, color = "orange", style = "arc"}}
- `group_boxes`: list of {{from = 0, to = 2, color = "gold"}}
- `message`: optional string describing the match

### Available Colors
purple, blue, cyan, orange, coral, gold, salmon, magenta, yellow, teal, red, gray

### Connector Styles
arc, line, dashed, bracket, arrow

## Helper Functions (with examples)

### Digit Analysis
- `count_digits(s)` → table of digit counts
  Example: `count_digits("11234111")` → `{{"1"=5, "2"=1, "3"=1, "4"=1}}`

- `unique_count(s)` → number of unique digits
  Example: `unique_count("11223344")` → `4`

- `digit_sum(s)` → sum of all digits
  Example: `digit_sum("12345678")` → `36`

- `most_common(s)` → digit, count (returns the most frequent digit)
  Example: `most_common("11123444")` → `"1", 3` (or `"4", 3`)

- `get_unique_digits(s)` → sorted string of unique digits
  Example: `get_unique_digits("11223344")` → `"1234"`

### Run/Pair Detection
- `find_runs(s)` → list of {{digit, start, length}} for ALL consecutive runs (including length=1)
  Example: `find_runs("11123345")` → `{{{{digit="1", start=0, length=3}}, {{digit="2", start=3, length=1}}, ...}}`
  **NOTE:** Returns every digit as a "run", even singles. Filter by length if needed.

- `find_consecutive_pairs(s)` → list of {{digit, start, length}} for pairs only (length=2)
  Example: `find_consecutive_pairs("11233455")` → `{{{{digit="1", start=0, length=2}}, {{digit="3", start=3, length=2}}, {{digit="5", start=6, length=2}}}}`
  **NOTE:** Each pair occupies 2 positions. Gap between pairs[1].start=0 and pairs[2].start=3 is 3, not 2.

- `find_pairs(s)` → list of {{digit, start}} - NO length field, just position
  Example: `find_pairs("11223344")` → `{{{{digit="1", start=0}}, {{digit="2", start=2}}, {{digit="3", start=4}}, {{digit="4", start=6}}}}`
  **NOTE:** Different from find_consecutive_pairs - returns NO length field.

- `count_pairs(s)` → count based on DIGIT FREQUENCY, not consecutive pairs
  Example: `count_pairs("12341234")` → `4` (each digit 2x, though no consecutive pairs)
  **WARNING:** NOT the same as #find_consecutive_pairs(s)!

- `has_four_consecutive_pairs(s)` → true ONLY for exact AABBCCDD (pairs at 0,2,4,6)
  **WARNING:** Returns false for "XAABBCCD" - pairs must start at position 0

- `find_triples(s)` → list of {{digit, start, length}} for 3+ consecutive identical
- `find_quads(s)` → list of {{digit, start, length}} for 4+ consecutive identical

- `has_n_consecutive(s, n)` → {{found, digit, start, length}} or **nil**
  **WARNING:** Returns nil if not found. Check `if result then` before accessing fields.

### Pattern Checks
- `is_palindrome(s)` → true if s equals its reverse
- `is_broken_palindrome(s, max_mismatches)` → {{matched, mismatches, positions}} or **nil**
  Returns positions as list of {{left_pos, right_pos}} pairs (0-indexed)
  **WARNING:** Returns nil if no mismatches or too many. Check before accessing.
- `is_ladder(s)` → true if ascending OR descending sequence
- `is_ascending(s)` / `is_descending(s)` → true for that direction only
- `find_ladder_of_length(s, min_len)` → {{found, start, length, ascending}} or **nil**
  **WARNING:** Returns nil if not found. Check `if result then` before accessing.
- `is_repeater(s)` → true if ABCDABCD pattern (first 4 = last 4)
- `is_super_repeater(s)` → true if ABABABAB pattern
- `is_alternating(s)` → true if XYXYXYXY pattern
- `is_bookended(s, n)` → true/false only (no position info)
  **NOTE:** To highlight bookends, use positions 0 to n-1 and 8-n to 7 yourself

### String Checks
- `only_digits(s, allowed)` → true if s contains only digits in allowed string
  Example: `only_digits("01010101", "01")` → `true`
- `starts_with(s, prefix)` / `ends_with(s, suffix)` / `contains(s, substr)`
- `all_flip_valid(s)` → true if all digits are in {{0,1,6,8,9}}
- `flip_string(s)` → 180° rotation ("6" ↔ "9", reversed) or **nil** if invalid
  **WARNING:** Returns nil if any digit isn't flip-valid. Check before using.

### Position Helpers
- `find_digit_positions(s, digit)` → list of 0-indexed positions
  Example: `find_digit_positions("12131415", "1")` → `{{0, 2, 4, 6}}`

### Visualization Helpers
- `highlight(positions, color, label)` → highlight table
- `highlight_range(start, stop, color, label)` → highlight for consecutive range
- `connector(from, to, color, style)` → connector table

## Important Rules

1. **Position indexing:** All highlight/connector positions are 0-indexed (0-7), but Lua strings are 1-indexed. Use `ctx.digits:sub(i+1, i+1)` to get character at position `i`.

2. **ctx.digit_list:** This is 1-indexed: `ctx.digit_list[1]` is the first digit.

3. **Use helper functions:** Don't reinvent - use `is_palindrome()`, `find_runs()`, `count_digits()`, `only_digits()`, etc.

4. **highlights vs group_boxes:** Use `highlights` for individual positions, `group_boxes` for a range of consecutive digits. For group_boxes, `from` and `to` are both inclusive 0-indexed positions.

5. **Always include Examples:** The Examples field in the header is required for the preview generator.

6. **Pair/run spacing math:** Each pair occupies 2 positions. For pattern AAXBBXCC: pair1 at 0-1, separator at 2, pair2 at 3-4, separator at 5, pair3 at 6-7. So pairs[1].start=0, pairs[2].start=3, pairs[3].start=6. The gap between consecutive pairs with a separator is 3 positions, not 2.

7. **NEVER shadow Lua built-ins:** Do NOT use variable names that shadow Lua's built-in functions. Avoid these names for variables: `pairs`, `ipairs`, `next`, `type`, `string`, `table`, `math`, `tonumber`, `tostring`, `print`, `error`, `select`, `unpack`. Use descriptive names like `pair_list`, `digit_pairs`, `found_pairs` instead of `pairs`.

## Your Task

Create a complete Lua pattern script for the following:

**Description:** {description}
{f"**Suggested Name:** {pattern_name}" if pattern_name else ""}
{examples_section}

## Response Format

Respond with ONLY the complete Lua script:
1. Start with the `--[[` header block containing Pattern, Description, Tier, and **Examples** (REQUIRED)
2. Include the `function match(ctx)` implementation
3. End with `end`

**REMINDER: You MUST include the Examples line in the header.** Example format:
```
Examples: ["11223456", "00112233", "12334455"]
```

Do not include any explanation before or after the code.
'''
        return prompt

    def _load_helpers_reference(self) -> str:
        """Load helper function signatures from helpers.lua."""
        helpers_path = Path(__file__).parent.parent / "patterns" / "lib" / "helpers.lua"
        if not helpers_path.exists():
            return "(helpers.lua not found)"

        try:
            content = helpers_path.read_text(encoding='utf-8')
        except Exception as e:
            return f"(Error reading helpers.lua: {e})"

        # Parse functions with their comments
        functions = []
        lines = content.split('\n')
        current_comment = []

        for line in lines:
            stripped = line.strip()

            # Collect comments
            if stripped.startswith('--') and not stripped.startswith('--[['):
                comment_text = stripped[2:].strip()
                if comment_text:
                    current_comment.append(comment_text)
            elif stripped.startswith('function ') and '(' in stripped:
                # Function definition
                match = re.match(r'function\s+(\w+)\s*\(([^)]*)\)', stripped)
                if match:
                    func_name = match.group(1)
                    params = match.group(2)
                    doc = ' '.join(current_comment) if current_comment else ''
                    functions.append(f"- `{func_name}({params})`: {doc}" if doc else f"- `{func_name}({params})`")
                current_comment = []
            elif stripped and not stripped.startswith('--'):
                current_comment = []

        return '\n'.join(functions)

    def _call_anthropic(self, prompt: str, progress_callback: Callable[[str], None] = None) -> AIGenerationResult:
        """Call the Anthropic API."""
        try:
            import anthropic
        except ImportError:
            return AIGenerationResult(
                success=False,
                error="The 'anthropic' package is not installed.\nRun: pip install anthropic"
            )

        if progress_callback:
            progress_callback("Connecting to Anthropic API...")

        try:
            client = anthropic.Anthropic(api_key=self.api_key)

            if progress_callback:
                progress_callback(f"Generating pattern with {self.model}...")

            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )

            raw_response = response.content[0].text
            lua_code = self._extract_lua_code(raw_response)

            if not lua_code:
                return AIGenerationResult(
                    success=False,
                    error="Could not extract Lua code from response",
                    raw_response=raw_response
                )

            return AIGenerationResult(
                success=True,
                lua_code=lua_code,
                raw_response=raw_response
            )

        except anthropic.AuthenticationError:
            return AIGenerationResult(success=False, error="Invalid API key")
        except anthropic.RateLimitError:
            return AIGenerationResult(success=False, error="Rate limit exceeded. Please try again later.")
        except Exception as e:
            return AIGenerationResult(success=False, error=str(e))

    def _call_openai(self, prompt: str, progress_callback: Callable[[str], None] = None) -> AIGenerationResult:
        """Call the OpenAI API."""
        try:
            import openai
        except ImportError:
            return AIGenerationResult(
                success=False,
                error="The 'openai' package is not installed.\nRun: pip install openai"
            )

        if progress_callback:
            progress_callback("Connecting to OpenAI API...")

        try:
            client = openai.OpenAI(api_key=self.api_key)

            if progress_callback:
                progress_callback(f"Generating pattern with {self.model}...")

            response = client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )

            raw_response = response.choices[0].message.content
            lua_code = self._extract_lua_code(raw_response)

            if not lua_code:
                return AIGenerationResult(
                    success=False,
                    error="Could not extract Lua code from response",
                    raw_response=raw_response
                )

            return AIGenerationResult(
                success=True,
                lua_code=lua_code,
                raw_response=raw_response
            )

        except openai.AuthenticationError:
            return AIGenerationResult(success=False, error="Invalid API key")
        except openai.RateLimitError:
            return AIGenerationResult(success=False, error="Rate limit exceeded. Please try again later.")
        except Exception as e:
            return AIGenerationResult(success=False, error=str(e))

    def _extract_lua_code(self, response: str) -> str:
        """Extract Lua code from the AI response."""
        # Try to find code in markdown code blocks first
        # Match ```lua ... ``` or ``` ... ```
        code_block_pattern = r'```(?:lua)?\s*\n(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            # Return the longest match (most likely to be the full script)
            return max(matches, key=len).strip()

        # If no code blocks, try to find the script directly
        # Look for --[[ header and function match(ctx)
        if '--[[' in response and 'function match(ctx)' in response:
            # Find the start of the header
            start = response.find('--[[')
            # Find the end - look for the last 'end' that closes the function
            # This is tricky, so we'll just take everything from --[[ onwards
            code = response[start:].strip()

            # Try to find where the code ends (before any explanation)
            # Look for common patterns that indicate end of code
            end_markers = ['\n\n##', '\n\nThis ', '\n\nThe ', '\n\nNote:', '\n\nExplanation:']
            for marker in end_markers:
                if marker in code:
                    code = code[:code.find(marker)].strip()
                    break

            return code

        # Last resort: return the whole response if it looks like Lua
        if 'function match(ctx)' in response:
            return response.strip()

        return ""

    def _ensure_examples(self, lua_code: str, should_match: list[str] = None) -> str:
        """Ensure the Lua code has an Examples line in the header.

        If Examples is missing and user provided should_match examples,
        inject them into the header.
        """
        # Check if Examples already exists
        if 'Examples:' in lua_code or 'Examples :' in lua_code:
            return lua_code

        # No examples in code - try to inject if user provided some
        if not should_match:
            return lua_code

        import json
        examples_line = f'Examples: {json.dumps(should_match[:5])}'

        # Find the header block and inject Examples before --]]
        if '--[[' in lua_code and '--]]' in lua_code:
            # Find the closing --]]
            close_pos = lua_code.find('--]]')
            if close_pos > 0:
                # Insert Examples line before --]]
                # Find the last newline before --]]
                last_newline = lua_code.rfind('\n', 0, close_pos)
                if last_newline > 0:
                    lua_code = (
                        lua_code[:last_newline + 1] +
                        examples_line + '\n' +
                        lua_code[last_newline + 1:]
                    )

        return lua_code
