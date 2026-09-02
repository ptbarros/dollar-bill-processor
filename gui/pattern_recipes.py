"""
Pattern Recipes - Recipe-based pattern creation for non-coders.

Each recipe type encapsulates a common pattern structure with configurable
parameters, and can generate both Lua code and example serials.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import json
import random


@dataclass
class ParameterDef:
    """Definition for a dynamic parameter in the wizard UI."""
    name: str
    label: str
    widget_type: str  # 'dropdown', 'spinbox', 'checkbox_group', 'radio'
    options: list = field(default_factory=list)  # for dropdown/radio/checkbox_group
    min_value: int = None  # for spinbox
    max_value: int = None  # for spinbox
    default: Any = None
    description: str = ""  # tooltip or help text


class PatternRecipe(ABC):
    """Base class for pattern recipes."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Internal recipe name."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """User-friendly recipe name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Brief description of what this recipe creates."""
        ...

    @abstractmethod
    def get_parameter_definitions(self) -> list[ParameterDef]:
        """Return list of parameter definitions for the UI."""
        ...

    @abstractmethod
    def generate_lua(self, params: dict, pattern_name: str, pattern_desc: str,
                     tier: int, color: str = "orange") -> str:
        """Generate complete Lua script from parameters."""
        ...

    @abstractmethod
    def generate_examples(self, params: dict) -> list[str]:
        """Generate example serials that match the pattern."""
        ...


class LadderRecipe(PatternRecipe):
    """Recipe for ladder/sequence patterns (ascending or descending runs)."""

    @property
    def name(self) -> str:
        return "ladder"

    @property
    def display_name(self) -> str:
        return "Ladder/Sequence"

    @property
    def description(self) -> str:
        return "Find ascending or descending runs of consecutive digits"

    def get_parameter_definitions(self) -> list[ParameterDef]:
        return [
            ParameterDef(
                name="min_length",
                label="Minimum length",
                widget_type="dropdown",
                options=["4", "5", "6", "7", "8"],
                default="4",
                description="Minimum number of consecutive digits in sequence"
            ),
            ParameterDef(
                name="direction",
                label="Direction",
                widget_type="dropdown",
                options=["Either", "Ascending", "Descending"],
                default="Either",
                description="Match only ascending (1234), only descending (4321), or either"
            ),
        ]

    def generate_lua(self, params: dict, pattern_name: str, pattern_desc: str,
                     tier: int, color: str = "cyan") -> str:
        min_length = int(params.get("min_length", 4))
        direction = params.get("direction", "Either")
        examples = self.generate_examples(params)

        # Build direction check code
        if direction == "Either":
            direction_check = ""
        elif direction == "Ascending":
            direction_check = "\n    if not result.ascending then return {matched = false} end"
        else:  # Descending
            direction_check = "\n    if result.ascending then return {matched = false} end"

        # Build direction message
        if direction == "Either":
            dir_msg = "result.ascending and 'ascending' or 'descending'"
        elif direction == "Ascending":
            dir_msg = "'ascending'"
        else:
            dir_msg = "'descending'"

        script = f'''--[[
Pattern: {pattern_name}
Description: {pattern_desc}
Tier: {tier}
Examples: {json.dumps(examples)}
--]]

function match(ctx)
    local result = find_ladder_of_length(ctx.digits, {min_length}){direction_check}
    if not result then return {{matched = false}} end

    -- Build highlights for the ladder positions
    local positions = {{}}
    for i = 0, result.length - 1 do
        table.insert(positions, result.start + i)
    end

    local dir_text = ({dir_msg})

    return {{
        matched = true,
        highlights = {{{{positions = positions, color = "{color}"}}}},
        connectors = {{{{from = result.start, to = result.start + result.length - 1, color = "{color}", style = "arc"}}}},
        message = "Ladder of " .. result.length .. " digits (" .. dir_text .. ")"
    }}
end
'''
        return script

    def generate_examples(self, params: dict) -> list[str]:
        min_length = int(params.get("min_length", 4))
        direction = params.get("direction", "Either")
        examples = []

        # Generate ascending examples
        if direction in ("Either", "Ascending"):
            for start in range(10 - min_length):
                ladder = "".join(str(d) for d in range(start, start + min_length))
                if len(ladder) < 8:
                    # Pad with zeros
                    examples.append(ladder + "0" * (8 - len(ladder)))
                else:
                    examples.append(ladder[:8])
                if len(examples) >= 3:
                    break

        # Generate descending examples
        if direction in ("Either", "Descending"):
            for start in range(9, min_length - 1, -1):
                ladder = "".join(str(d) for d in range(start, start - min_length, -1))
                if len(ladder) < 8:
                    examples.append("0" * (8 - len(ladder)) + ladder)
                else:
                    examples.append(ladder[:8])
                if len(examples) >= 5:
                    break

        return examples[:5]


class DigitSetRecipe(PatternRecipe):
    """Recipe for patterns matching only specific digits."""

    @property
    def name(self) -> str:
        return "digit_set"

    @property
    def display_name(self) -> str:
        return "Digit Set Restriction"

    @property
    def description(self) -> str:
        return "Match serials using only specific digits"

    def get_parameter_definitions(self) -> list[ParameterDef]:
        return [
            ParameterDef(
                name="digits",
                label="Allowed digits",
                widget_type="checkbox_group",
                options=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                default=["0", "1"],  # Binary default
                description="Select which digits are allowed in matching serials"
            ),
        ]

    def generate_lua(self, params: dict, pattern_name: str, pattern_desc: str,
                     tier: int, color: str = "blue") -> str:
        digits = params.get("digits", ["0", "1"])
        if isinstance(digits, str):
            digits = list(digits)
        allowed = "".join(sorted(digits))
        examples = self.generate_examples(params)

        # Choose color based on digit set
        if set(digits) == {"0", "1"}:
            color = "blue"  # Binary
        elif set(digits) == {"0", "1", "6", "8", "9"}:
            color = "purple"  # Flipper
        elif all(int(d) % 2 == 0 for d in digits):
            color = "cyan"  # Evens
        elif all(int(d) % 2 == 1 for d in digits):
            color = "magenta"  # Odds

        script = f'''--[[
Pattern: {pattern_name}
Description: {pattern_desc}
Tier: {tier}
Examples: {json.dumps(examples)}
--]]

function match(ctx)
    if not only_digits(ctx.digits, "{allowed}") then
        return {{matched = false}}
    end

    -- Highlight all positions
    local positions = {{0, 1, 2, 3, 4, 5, 6, 7}}

    return {{
        matched = true,
        highlights = {{{{positions = positions, color = "{color}"}}}},
        message = "Uses only digits: {allowed}"
    }}
end
'''
        return script

    def generate_examples(self, params: dict) -> list[str]:
        digits = params.get("digits", ["0", "1"])
        if isinstance(digits, str):
            digits = list(digits)

        examples = []
        # Generate random combinations
        for _ in range(10):
            serial = "".join(random.choice(digits) for _ in range(8))
            if serial not in examples:
                examples.append(serial)
            if len(examples) >= 5:
                break

        # Ensure at least some variety
        if len(digits) >= 2:
            # Alternating pattern
            alt = (digits[0] + digits[1]) * 4
            if alt not in examples:
                examples.insert(0, alt)

        return examples[:5]


class RepeatingRecipe(PatternRecipe):
    """Recipe for repeating pattern types."""

    @property
    def name(self) -> str:
        return "repeating"

    @property
    def display_name(self) -> str:
        return "Repeating Patterns"

    @property
    def description(self) -> str:
        return "Match specific repetition structures"

    def get_parameter_definitions(self) -> list[ParameterDef]:
        return [
            ParameterDef(
                name="type",
                label="Pattern type",
                widget_type="dropdown",
                options=[
                    "Consecutive Pairs (AABBCCDD)",
                    "Repeater (ABCDABCD)",
                    "Super Repeater (ABABABAB)",
                    "Alternator (XYXYXYXY)",
                ],
                default="Consecutive Pairs (AABBCCDD)",
                description="Select the repetition structure to match"
            ),
        ]

    def generate_lua(self, params: dict, pattern_name: str, pattern_desc: str,
                     tier: int, color: str = "teal") -> str:
        pattern_type = params.get("type", "Consecutive Pairs (AABBCCDD)")
        examples = self.generate_examples(params)

        if "Consecutive Pairs" in pattern_type:
            check_func = "has_four_consecutive_pairs"
            message = "Four consecutive pairs (AABBCCDD)"
            highlight_code = '''    -- Highlight each pair with alternating colors
    local colors = {"teal", "coral", "gold", "salmon"}
    local highlights = {}
    local group_boxes = {}
    for i = 0, 3 do
        local start = i * 2
        table.insert(group_boxes, {from = start, to = start + 1, color = colors[i+1]})
    end'''
            return_extras = "group_boxes = group_boxes,"

        elif "Super Repeater" in pattern_type:
            check_func = "is_super_repeater"
            message = "Super repeater (ABABABAB)"
            color = "magenta"
            highlight_code = '''    -- Highlight the repeating pair pattern
    local pair = ctx.digits:sub(1, 2)
    local positions_a = {0, 2, 4, 6}
    local positions_b = {1, 3, 5, 7}
    local highlights = {
        {positions = positions_a, color = "magenta"},
        {positions = positions_b, color = "coral"}
    }
    local group_boxes = {}'''
            return_extras = ""

        elif "Repeater" in pattern_type:
            check_func = "is_repeater"
            message = "Repeater (ABCDABCD)"
            color = "magenta"
            highlight_code = '''    -- Highlight first and second halves
    local highlights = {}
    local group_boxes = {
        {from = 0, to = 3, color = "magenta"},
        {from = 4, to = 7, color = "coral"}
    }
    local connectors = {}
    for i = 0, 3 do
        table.insert(connectors, {from = i, to = i + 4, color = "magenta", style = "arc"})
    end'''
            return_extras = "connectors = connectors, group_boxes = group_boxes,"

        else:  # Alternator
            check_func = "is_alternating"
            message = "Alternating pattern (XYXYXYXY)"
            color = "cyan"
            highlight_code = '''    -- Highlight alternating digits
    local d1 = ctx.digits:sub(1, 1)
    local d2 = ctx.digits:sub(2, 2)
    local highlights = {
        {positions = {0, 2, 4, 6}, color = "cyan"},
        {positions = {1, 3, 5, 7}, color = "coral"}
    }
    local group_boxes = {}'''
            return_extras = ""

        script = f'''--[[
Pattern: {pattern_name}
Description: {pattern_desc}
Tier: {tier}
Examples: {json.dumps(examples)}
--]]

function match(ctx)
    if not {check_func}(ctx.digits) then
        return {{matched = false}}
    end

{highlight_code}

    return {{
        matched = true,
        highlights = highlights,
        {return_extras}
        message = "{message}"
    }}
end
'''
        return script

    def generate_examples(self, params: dict) -> list[str]:
        pattern_type = params.get("type", "Consecutive Pairs (AABBCCDD)")
        examples = []

        if "Consecutive Pairs" in pattern_type:
            # AABBCCDD patterns
            examples = ["11223344", "22334455", "00112233", "33445566", "55667788"]

        elif "Super Repeater" in pattern_type:
            # ABABABAB patterns
            examples = ["12121212", "34343434", "56565656", "01010101", "78787878"]

        elif "Repeater" in pattern_type:
            # ABCDABCD patterns
            examples = ["12341234", "56785678", "01230123", "98769876", "11221122"]

        else:  # Alternator
            # XYXYXYXY patterns (same as super repeater for distinct digits)
            examples = ["12121212", "34343434", "56565656", "01010101", "89898989"]

        return examples[:5]


class PalindromeRecipe(PatternRecipe):
    """Recipe for palindrome/radar patterns."""

    @property
    def name(self) -> str:
        return "palindrome"

    @property
    def display_name(self) -> str:
        return "Palindrome/Radar"

    @property
    def description(self) -> str:
        return "Match serials that read the same forwards and backwards"

    def get_parameter_definitions(self) -> list[ParameterDef]:
        return [
            ParameterDef(
                name="strict",
                label="Match type",
                widget_type="dropdown",
                options=["Exact palindrome", "Allow 1 mismatch", "Allow 2 mismatches"],
                default="Exact palindrome",
                description="Exact requires perfect palindrome, mismatches allow near-palindromes"
            ),
        ]

    def generate_lua(self, params: dict, pattern_name: str, pattern_desc: str,
                     tier: int, color: str = "orange") -> str:
        match_type = params.get("strict", "Exact palindrome")
        examples = self.generate_examples(params)

        if match_type == "Exact palindrome":
            check_code = '''    if not is_palindrome(ctx.digits) then
        return {matched = false}
    end

    -- Highlight paired positions
    local colors = {"orange", "coral", "gold", "salmon"}
    local highlights = {}
    local connectors = {}

    for i = 0, 3 do
        local j = 7 - i
        table.insert(highlights, {positions = {i, j}, color = colors[i+1]})
        table.insert(connectors, {from = i, to = j, color = colors[i+1], style = "arc"})
    end'''
            message = "Perfect palindrome"

        else:
            max_mismatches = 1 if "1 mismatch" in match_type else 2
            check_code = f'''    -- Check for exact palindrome first
    if is_palindrome(ctx.digits) then
        return {{matched = false}}  -- Exact palindromes handled by stricter pattern
    end

    local result = is_broken_palindrome(ctx.digits, {max_mismatches})
    if not result then
        return {{matched = false}}
    end

    -- Highlight matching pairs in green, mismatches in red
    local highlights = {{}}
    local connectors = {{}}

    for i = 0, 3 do
        local j = 7 - i
        local is_mismatch = false
        for _, pos in ipairs(result.positions) do
            if pos[1] == i then is_mismatch = true break end
        end

        local col = is_mismatch and "red" or "orange"
        table.insert(highlights, {{positions = {{i, j}}, color = col}})
        table.insert(connectors, {{from = i, to = j, color = col, style = "arc"}})
    end'''
            message = f"Near palindrome ({max_mismatches} mismatch)"

        script = f'''--[[
Pattern: {pattern_name}
Description: {pattern_desc}
Tier: {tier}
Examples: {json.dumps(examples)}
--]]

function match(ctx)
{check_code}

    return {{
        matched = true,
        highlights = highlights,
        connectors = connectors,
        message = "{message}"
    }}
end
'''
        return script

    def generate_examples(self, params: dict) -> list[str]:
        match_type = params.get("strict", "Exact palindrome")

        if match_type == "Exact palindrome":
            return ["12344321", "45677654", "11111111", "12321232", "98766789"]
        else:
            # Near palindromes with 1-2 mismatches
            return ["12345321", "45677654", "12344320", "98766780", "11211121"]


class DigitSumRecipe(PatternRecipe):
    """Recipe for digit sum patterns."""

    @property
    def name(self) -> str:
        return "digit_sum"

    @property
    def display_name(self) -> str:
        return "Digit Sum"

    @property
    def description(self) -> str:
        return "Match serials where digits add up to a specific value"

    def get_parameter_definitions(self) -> list[ParameterDef]:
        return [
            ParameterDef(
                name="mode",
                label="Match mode",
                widget_type="dropdown",
                options=["Exact value", "Range"],
                default="Exact value",
                description="Match exact sum or a range of sums"
            ),
            ParameterDef(
                name="target",
                label="Target sum",
                widget_type="spinbox",
                min_value=0,
                max_value=72,  # 8 * 9 = 72 max
                default=7,
                description="Target sum value (for exact mode)"
            ),
            ParameterDef(
                name="min_sum",
                label="Minimum sum",
                widget_type="spinbox",
                min_value=0,
                max_value=72,
                default=0,
                description="Minimum sum (for range mode)"
            ),
            ParameterDef(
                name="max_sum",
                label="Maximum sum",
                widget_type="spinbox",
                min_value=0,
                max_value=72,
                default=10,
                description="Maximum sum (for range mode)"
            ),
        ]

    def generate_lua(self, params: dict, pattern_name: str, pattern_desc: str,
                     tier: int, color: str = "gold") -> str:
        mode = params.get("mode", "Exact value")
        target = int(params.get("target", 7))
        min_sum = int(params.get("min_sum", 0))
        max_sum = int(params.get("max_sum", 10))
        examples = self.generate_examples(params)

        if mode == "Exact value":
            check_code = f'''    local sum = digit_sum(ctx.digits)
    if sum ~= {target} then
        return {{matched = false}}
    end'''
            message = f"Digit sum equals {target}"
        else:
            check_code = f'''    local sum = digit_sum(ctx.digits)
    if sum < {min_sum} or sum > {max_sum} then
        return {{matched = false}}
    end'''
            message = f"Digit sum between {min_sum} and {max_sum}"

        script = f'''--[[
Pattern: {pattern_name}
Description: {pattern_desc}
Tier: {tier}
Examples: {json.dumps(examples)}
--]]

function match(ctx)
{check_code}

    return {{
        matched = true,
        highlights = {{{{positions = {{0, 1, 2, 3, 4, 5, 6, 7}}, color = "{color}"}}}},
        message = "{message} (sum = " .. digit_sum(ctx.digits) .. ")"
    }}
end
'''
        return script

    def generate_examples(self, params: dict) -> list[str]:
        mode = params.get("mode", "Exact value")
        target = int(params.get("target", 7))
        min_sum = int(params.get("min_sum", 0))
        max_sum = int(params.get("max_sum", 10))

        examples = []

        if mode == "Exact value":
            # Generate serials with the target sum
            targets = [target]
        else:
            # Generate serials in the range
            targets = list(range(min_sum, min(max_sum + 1, min_sum + 5)))

        for t in targets:
            serial = self._generate_serial_with_sum(t)
            if serial and serial not in examples:
                examples.append(serial)

        return examples[:5]

    def _generate_serial_with_sum(self, target: int) -> Optional[str]:
        """Generate an 8-digit serial with the given digit sum."""
        if target < 0 or target > 72:
            return None

        # Simple approach: fill digits greedily
        digits = []
        remaining = target
        for i in range(8):
            positions_left = 8 - i - 1
            max_remaining = positions_left * 9
            # Take as much as we can while leaving room
            take = min(9, remaining - 0)  # At least 0 for remaining positions
            take = max(0, min(take, remaining))  # Clamp to [0, remaining]
            if remaining - take > max_remaining:
                take = remaining - max_remaining
            digits.append(str(take))
            remaining -= take

        if remaining != 0:
            return None

        return "".join(digits)


class BookendRecipe(PatternRecipe):
    """Recipe for bookend patterns (first N = last N)."""

    @property
    def name(self) -> str:
        return "bookend"

    @property
    def display_name(self) -> str:
        return "Bookends"

    @property
    def description(self) -> str:
        return "Match serials where first N digits equal last N digits"

    def get_parameter_definitions(self) -> list[ParameterDef]:
        return [
            ParameterDef(
                name="length",
                label="Bookend length",
                widget_type="dropdown",
                options=["1", "2", "3", "4"],
                default="2",
                description="Number of digits to match at start and end"
            ),
        ]

    def generate_lua(self, params: dict, pattern_name: str, pattern_desc: str,
                     tier: int, color: str = "orange") -> str:
        length = int(params.get("length", 2))
        examples = self.generate_examples(params)

        script = f'''--[[
Pattern: {pattern_name}
Description: {pattern_desc}
Tier: {tier}
Examples: {json.dumps(examples)}
--]]

function match(ctx)
    if not is_bookended(ctx.digits, {length}) then
        return {{matched = false}}
    end

    -- Highlight the matching bookends
    local group_boxes = {{
        {{from = 0, to = {length - 1}, color = "{color}"}},
        {{from = {8 - length}, to = 7, color = "{color}"}}
    }}

    local connectors = {{
        {{from = 0, to = {8 - length}, color = "{color}", style = "arc"}}
    }}

    return {{
        matched = true,
        group_boxes = group_boxes,
        connectors = connectors,
        message = "Bookended with {length} digit(s): " .. ctx.digits:sub(1, {length})
    }}
end
'''
        return script

    def generate_examples(self, params: dict) -> list[str]:
        length = int(params.get("length", 2))
        examples = []

        # Generate examples with matching bookends
        bookends = ["12", "34", "56", "00", "99"][:5] if length == 2 else \
                   ["1", "3", "5", "0", "9"][:5] if length == 1 else \
                   ["123", "456", "000", "999", "321"][:5] if length == 3 else \
                   ["1234", "5678", "0000", "9999", "4321"][:5]

        for be in bookends:
            be = be[:length]  # Ensure correct length
            middle_len = 8 - 2 * length
            middle = "0" * middle_len
            serial = be + middle + be
            if len(serial) == 8:
                examples.append(serial)

        return examples[:5]


# Registry of all available recipes
RECIPE_REGISTRY: list[PatternRecipe] = [
    LadderRecipe(),
    DigitSetRecipe(),
    RepeatingRecipe(),
    PalindromeRecipe(),
    DigitSumRecipe(),
    BookendRecipe(),
]


def get_recipe_by_name(name: str) -> Optional[PatternRecipe]:
    """Get a recipe by its internal name."""
    for recipe in RECIPE_REGISTRY:
        if recipe.name == name:
            return recipe
    return None


def get_all_recipes() -> list[PatternRecipe]:
    """Get all available recipes."""
    return RECIPE_REGISTRY
