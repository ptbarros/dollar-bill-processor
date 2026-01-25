"""
Pattern Engine v2 - Design Sketch

This engine evaluates patterns defined in patterns_v2.yaml
Single source of truth, GUI-friendly, multi-denomination support.
"""

import re
import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class PatternMatch:
    """Result of a pattern match."""
    name: str
    description: str
    tier: int


class PatternEngine:
    """
    Evaluates serial numbers against pattern definitions.

    All pattern logic is driven by the YAML config - no hardcoded patterns.
    """

    def __init__(self, config_path: Path = None):
        self.config_path = config_path or Path("patterns_v2.yaml")
        self.config = self._load_config()
        self.patterns = self._build_patterns()

    def _load_config(self) -> dict:
        """Load pattern definitions from YAML."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _build_patterns(self) -> dict:
        """Build pattern lookup from config."""
        patterns = {}

        # Load main patterns
        for name, defn in self.config.get('patterns', {}).items():
            if defn.get('enabled', True):
                patterns[name] = defn

        # Load custom patterns
        custom = self.config.get('custom_patterns') or {}
        for name, defn in custom.items():
            if defn and defn.get('enabled', True):
                patterns[name] = defn

        return patterns

    def reload(self):
        """Reload patterns from config file (for GUI live updates)."""
        self.config = self._load_config()
        self.patterns = self._build_patterns()

    # -------------------------------------------------------------------------
    # BUILT-IN CHECK FUNCTIONS
    # -------------------------------------------------------------------------

    def _check_solid(self, digits: str) -> bool:
        """All digits are the same."""
        return len(set(digits)) == 1

    def _check_palindrome(self, digits: str) -> bool:
        """Reads same forwards and backwards."""
        return digits == digits[::-1]

    def _check_ladder_asc(self, digits: str) -> bool:
        """Perfect ascending ladder (01234567)."""
        return digits in "0123456789"

    def _check_ladder_desc(self, digits: str) -> bool:
        """Perfect descending ladder (76543210)."""
        return digits in "9876543210"

    def _check_all_even(self, digits: str) -> bool:
        """All digits are even."""
        return all(int(d) % 2 == 0 for d in digits)

    def _check_all_odd(self, digits: str) -> bool:
        """All digits are odd."""
        return all(int(d) % 2 == 1 for d in digits)

    def _check_repeater(self, digits: str) -> bool:
        """Two-digit pattern repeated (ABABABAB)."""
        if len(digits) != 8:
            return False
        pair = digits[:2]
        return digits == pair * 4

    # Map of check names to functions
    CHECK_FUNCTIONS = {
        'solid': _check_solid,
        'palindrome': _check_palindrome,
        'ladder_asc': _check_ladder_asc,
        'ladder_desc': _check_ladder_desc,
        'all_even': _check_all_even,
        'all_odd': _check_all_odd,
        'repeater': _check_repeater,
    }

    # -------------------------------------------------------------------------
    # RULE EVALUATION
    # -------------------------------------------------------------------------

    def _evaluate_rule(self, rule_type: str, rule_value, digits: str, full_serial: str) -> bool:
        """Evaluate a single rule against the serial number."""

        if rule_type == 'regex':
            return bool(re.search(rule_value, digits))

        elif rule_type == 'contains':
            return rule_value in digits

        elif rule_type == 'starts_with':
            return digits.startswith(rule_value)

        elif rule_type == 'ends_with':
            # Can apply to digits or full serial
            return digits.endswith(rule_value) or full_serial.endswith(rule_value)

        elif rule_type == 'unique_count':
            return len(set(digits)) == rule_value

        elif rule_type == 'digit_sum':
            return sum(int(d) for d in digits) == rule_value

        elif rule_type == 'check':
            # Built-in check function
            check_fn = self.CHECK_FUNCTIONS.get(rule_value)
            if check_fn:
                return check_fn(self, digits)
            return False

        elif rule_type == 'all':
            # AND - all sub-rules must match
            return all(
                self._evaluate_rules(sub_rule, digits, full_serial)
                for sub_rule in rule_value
            )

        elif rule_type == 'any':
            # OR - any sub-rule must match
            return any(
                self._evaluate_rules(sub_rule, digits, full_serial)
                for sub_rule in rule_value
            )

        return False

    def _evaluate_rules(self, rules: dict, digits: str, full_serial: str) -> bool:
        """Evaluate a rules dict (can have multiple rule types)."""
        for rule_type, rule_value in rules.items():
            if not self._evaluate_rule(rule_type, rule_value, digits, full_serial):
                return False
        return True

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def extract_digits(self, serial: str) -> str:
        """Extract just the numeric portion of a serial number."""
        return ''.join(c for c in serial if c.isdigit())

    def classify(self, serial: str) -> list[PatternMatch]:
        """
        Classify a serial number and return all matching patterns.

        Args:
            serial: Full serial number (e.g., "A12345678B")

        Returns:
            List of PatternMatch objects for all matching patterns
        """
        if not serial:
            return []

        digits = self.extract_digits(serial)
        matches = []

        for name, defn in self.patterns.items():
            rules = defn.get('rules', {})

            # Some patterns use full serial (like STAR notes)
            if defn.get('uses_full_serial'):
                target = serial
            else:
                target = digits

            if self._evaluate_rules(rules, digits, serial):
                matches.append(PatternMatch(
                    name=name,
                    description=defn.get('description', ''),
                    tier=defn.get('tier', 5)
                ))

        # Sort by tier (lower = rarer = more important)
        matches.sort(key=lambda m: m.tier)

        return matches

    def classify_simple(self, serial: str) -> list[str]:
        """Return just the pattern names (for compatibility)."""
        return [m.name for m in self.classify(serial)]

    def get_pattern_info(self, name: str) -> Optional[dict]:
        """Get full info about a pattern (for GUI)."""
        return self.patterns.get(name)

    def get_all_patterns(self) -> dict:
        """Get all pattern definitions (for GUI)."""
        return self.patterns.copy()

    def set_pattern_enabled(self, name: str, enabled: bool):
        """Enable/disable a pattern (for GUI)."""
        if name in self.config.get('patterns', {}):
            self.config['patterns'][name]['enabled'] = enabled
        elif name in self.config.get('custom_patterns', {}):
            self.config['custom_patterns'][name]['enabled'] = enabled
        self.patterns = self._build_patterns()

    def save_config(self):
        """Save current config back to file (for GUI)."""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)


# -----------------------------------------------------------------------------
# EXAMPLE USAGE
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick test
    engine = PatternEngine(Path("patterns_v2_example.yaml"))

    test_serials = [
        "A88888888B",  # SOLID
        "A12344321B",  # RADAR
        "A12121212B",  # REPEATER
        "A11211121B",  # BINARY
        "A12345678B",  # LADDER
        "A00000123B",  # LOW_SERIAL
        "A12777456B",  # LUCKY_777
        "A12345678*",  # STAR
    ]

    print("Pattern Engine v2 - Test\n")
    print("=" * 60)

    for serial in test_serials:
        matches = engine.classify(serial)
        if matches:
            names = [m.name for m in matches]
            print(f"{serial}: {', '.join(names)}")
        else:
            print(f"{serial}: (no matches)")
