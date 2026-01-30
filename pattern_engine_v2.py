"""
Pattern Engine v2 - Complete Implementation

Evaluates serial numbers against patterns defined in patterns_v2.yaml.
Single source of truth for CLI, GUI, and multi-denomination support.
"""

import re
import yaml
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from collections import Counter


@dataclass
class PatternMatch:
    """Result of a pattern match."""
    name: str
    description: str
    tier: int


class PatternEngine:
    """
    Evaluates serial numbers against pattern definitions.
    All pattern logic driven by YAML config.
    """

    def __init__(self, config_path: Path = None):
        if config_path is None:
            self.config_path = Path(__file__).parent / "patterns_v2.yaml"
        else:
            self.config_path = Path(config_path) if isinstance(config_path, str) else config_path
        self.user_config_path = self.config_path.parent / "user_patterns.yaml"
        self.config = self._load_config()
        self.user_config = self._load_user_config()
        self.patterns = self._build_patterns()

    def _load_config(self) -> dict:
        """Load pattern definitions from YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Pattern config not found: {self.config_path}")
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_user_config(self) -> dict:
        """Load user-specific pattern settings (custom patterns, enable/disable overrides)."""
        if self.user_config_path.exists():
            with open(self.user_config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {'custom_patterns': {}, 'disabled_patterns': [], 'enabled_patterns': []}

    def _build_patterns(self) -> dict:
        """Build pattern lookup from config."""
        patterns = {}

        # Get user overrides
        disabled = set(self.user_config.get('disabled_patterns', []))
        enabled = set(self.user_config.get('enabled_patterns', []))

        # Get user rule overrides (e.g., custom thresholds for GAS_PUMP)
        pattern_overrides = self.user_config.get('pattern_overrides', {})

        # Load main patterns (with user enable/disable overrides)
        for name, defn in self.config.get('patterns', {}).items():
            if defn is None:
                continue
            # Check user override first, then default
            if name in disabled:
                continue
            if name in enabled or defn.get('enabled', True):
                # Apply any rule overrides from user config
                if name in pattern_overrides:
                    defn = defn.copy()  # Don't modify original
                    if 'rules' in defn:
                        defn['rules'] = defn['rules'].copy()
                        for rule_type, value in pattern_overrides[name].items():
                            defn['rules'][rule_type] = value
                patterns[name] = defn

        # Load custom patterns from main config (legacy support)
        custom = self.config.get('custom_patterns') or {}
        for name, defn in custom.items():
            if defn and defn.get('enabled', True) and name not in disabled:
                patterns[name] = defn

        # Load user custom patterns (takes precedence)
        user_custom = self.user_config.get('custom_patterns') or {}
        for name, defn in user_custom.items():
            if defn and defn.get('enabled', True):
                patterns[name] = defn

        return patterns

    def reload(self):
        """Reload patterns from config files."""
        self.config = self._load_config()
        self.user_config = self._load_user_config()
        self.patterns = self._build_patterns()

    # =========================================================================
    # BUILT-IN CHECK FUNCTIONS
    # =========================================================================

    def _check_solid(self, digits: str) -> bool:
        """All digits identical."""
        return len(digits) == 8 and len(set(digits)) == 1

    def _check_palindrome(self, digits: str) -> bool:
        """Reads same forwards and backwards."""
        return len(digits) == 8 and digits == digits[::-1]

    def _check_repeater(self, digits: str) -> bool:
        """First 4 digits repeat (ABCDABCD)."""
        return len(digits) == 8 and digits[:4] == digits[4:]

    def _check_ladder_asc(self, digits: str) -> bool:
        """Perfect ascending ladder."""
        if len(digits) != 8:
            return False
        nums = [int(d) for d in digits]
        return all(nums[i] + 1 == nums[i+1] for i in range(7))

    def _check_ladder_desc(self, digits: str) -> bool:
        """Perfect descending ladder."""
        if len(digits) != 8:
            return False
        nums = [int(d) for d in digits]
        return all(nums[i] - 1 == nums[i+1] for i in range(7))

    def _check_all_even(self, digits: str) -> bool:
        """All digits even."""
        return len(digits) == 8 and all(d in '02468' for d in digits)

    def _check_all_odd(self, digits: str) -> bool:
        """All digits odd."""
        return len(digits) == 8 and all(d in '13579' for d in digits)

    def _check_binary_digits(self, digits: str) -> bool:
        """Only 0s and 1s."""
        return set(digits).issubset({'0', '1'})

    def _check_alternator(self, digits: str) -> bool:
        """Alternating pattern ABABABAB."""
        if len(digits) != 8 or len(set(digits)) != 2:
            return False
        return all(digits[i] == digits[i % 2] for i in range(8))

    def _check_four_pairs(self, digits: str) -> bool:
        """Four consecutive pairs AABBCCDD."""
        if len(digits) != 8:
            return False
        return all(digits[i*2] == digits[i*2+1] for i in range(4))

    def _check_three_pairs(self, digits: str) -> bool:
        """Contains at least 3 pairs."""
        if len(digits) != 8:
            return False
        pair_count = 0
        i = 0
        while i < 7:
            if digits[i] == digits[i+1]:
                pair_count += 1
                i += 2
            else:
                i += 1
        return pair_count >= 3

    def _check_full_house(self, digits: str) -> bool:
        """5 of one digit, 3 of another."""
        if len(digits) != 8:
            return False
        counts = sorted(Counter(digits).values(), reverse=True)
        return counts == [5, 3]

    def _check_seven_of_kind(self, digits: str) -> bool:
        """7 of the same digit anywhere in the serial."""
        if len(digits) != 8:
            return False
        counts = Counter(digits)
        return any(c >= 7 for c in counts.values())

    def _check_two_pair_triple(self, digits: str) -> bool:
        """Triple + two pairs."""
        if len(digits) != 8:
            return False
        counts = sorted(Counter(digits).values(), reverse=True)
        return counts in [[3, 2, 2, 1], [3, 2, 2]]

    def _check_triple_double_double(self, digits: str) -> bool:
        """Triple + double + double + single."""
        if len(digits) != 8:
            return False
        counts = sorted(Counter(digits).values(), reverse=True)
        return counts == [3, 2, 2, 1]

    def _check_consecutive_triples(self, digits: str) -> bool:
        """Two triples back-to-back."""
        if len(digits) != 8:
            return False
        for i in range(3):
            if (digits[i] == digits[i+1] == digits[i+2] and
                digits[i+3] == digits[i+4] == digits[i+5] and
                digits[i] != digits[i+3]):
                return True
        return False

    def _check_chunky_ladder(self, digits: str) -> bool:
        """Paired ladder 11223344."""
        if len(digits) != 8:
            return False
        if not all(digits[i*2] == digits[i*2+1] for i in range(4)):
            return False
        nums = [int(digits[i*2]) for i in range(4)]
        return (all(nums[i] + 1 == nums[i+1] for i in range(3)) or
                all(nums[i] - 1 == nums[i+1] for i in range(3)))

    def _check_doubles_ladder(self, digits: str) -> bool:
        """Same as chunky_ladder."""
        return self._check_chunky_ladder(digits)

    def _check_flipper_digits(self, digits: str) -> bool:
        """Only flippable digits (0,1,6,8,9)."""
        return set(digits).issubset({'0', '1', '6', '8', '9'})

    def _check_true_flipper(self, digits: str) -> bool:
        """Reads same upside down."""
        if len(digits) != 8:
            return False
        flip_map = {'0': '0', '1': '1', '6': '9', '8': '8', '9': '6'}
        if not set(digits).issubset(set(flip_map.keys())):
            return False
        flipped = ''.join(flip_map[d] for d in reversed(digits))
        return digits == flipped

    def _check_near_flipper(self, digits: str) -> bool:
        """One digit from true flipper."""
        if len(digits) != 8:
            return False
        flip_map = {'0', '1', '6', '8', '9'}
        return sum(1 for d in digits if d not in flip_map) == 1

    def _check_broken_radar(self, digits: str) -> bool:
        """One digit from radar."""
        if len(digits) != 8:
            return False
        reversed_d = digits[::-1]
        return sum(1 for i in range(8) if digits[i] != reversed_d[i]) == 2

    def _check_sequential_trinary(self, digits: str) -> bool:
        """Trinary with sequential digits."""
        if len(digits) != 8:
            return False
        unique = sorted(set(digits))
        if len(unique) not in [2, 3]:
            return False
        nums = [int(d) for d in unique]
        return all(nums[i] + 1 == nums[i+1] for i in range(len(nums)-1))

    def _check_pyramid_ladder(self, digits: str) -> bool:
        """Up then down pattern."""
        if len(digits) < 5:
            return False
        nums = [int(d) for d in digits]
        for i in range(len(nums) - 4):
            seg = nums[i:i+5]
            if (seg[0] < seg[1] < seg[2] and seg[2] > seg[3] > seg[4] and
                seg[0] == seg[4] and seg[1] == seg[3]):
                return True
        return False

    def _check_counting_ladder(self, digits: str) -> bool:
        """Counting pattern 12123123."""
        if len(digits) != 8:
            return False
        return digits == '12123123' or (digits[:2] == '12' and digits[2:5] == '123' and digits[5:] == '123')

    def _check_step_ladder(self, digits: str) -> bool:
        """Steps of 2."""
        if len(digits) != 8:
            return False
        nums = [int(d) for d in digits]
        for i in range(5):
            if all(nums[i+j+1] - nums[i+j] == 2 for j in range(3)):
                return True
            if all(nums[i+j] - nums[i+j+1] == 2 for j in range(3)):
                return True
        return False

    def _check_super_ladder(self, digits: str) -> bool:
        """Two-digit increments 01020304."""
        if len(digits) != 8:
            return False
        pairs = [digits[i:i+2] for i in range(0, 8, 2)]
        try:
            nums = [int(p) for p in pairs]
            diff = nums[1] - nums[0]
            if diff != 0 and all(nums[i+1] - nums[i] == diff for i in range(3)):
                return True
        except:
            pass
        return False

    def _check_counting_step(self, digits: str, step: int) -> bool:
        """Counting ladder with given step."""
        if len(digits) != 8:
            return False
        pairs = [digits[i:i+2] for i in range(0, 8, 2)]
        try:
            nums = [int(p) for p in pairs]
            if all(nums[i+1] - nums[i] == step for i in range(3)):
                return True
            if all(nums[i] - nums[i+1] == step for i in range(3)):
                return True
        except:
            pass
        return False

    def _check_ladder_n(self, digits: str, length: int) -> bool:
        """Contains ladder of given length."""
        if len(digits) < length:
            return False
        nums = [int(d) for d in digits]
        for i in range(len(nums) - length + 1):
            seg = nums[i:i+length]
            if all(seg[j] + 1 == seg[j+1] for j in range(length-1)):
                return True
            if all(seg[j] - 1 == seg[j+1] for j in range(length-1)):
                return True
        return False

    def _check_quad_symmetry(self, digits: str) -> bool:
        """Internal quad symmetry for super radar."""
        if len(digits) != 8:
            return False
        return digits[:4] == digits[:4][::-1]

    def _check_ladder_and_quad(self, digits: str) -> bool:
        """Contains 4+ ladder AND quad."""
        if len(digits) != 8:
            return False
        has_quad = bool(re.search(r'(\d)\1{3}', digits))
        if not has_quad:
            return False
        return self._check_ladder_n(digits, 4)

    def _check_birthday(self, digits: str) -> bool:
        """Valid date format."""
        if len(digits) != 8:
            return False
        # MMDDYYYY
        try:
            mm, dd, yyyy = int(digits[:2]), int(digits[2:4]), int(digits[4:])
            if 1 <= mm <= 12 and 1 <= dd <= 31 and 1900 <= yyyy <= 2030:
                return True
        except:
            pass
        # DDMMYYYY
        try:
            dd, mm, yyyy = int(digits[:2]), int(digits[2:4]), int(digits[4:])
            if 1 <= mm <= 12 and 1 <= dd <= 31 and 1900 <= yyyy <= 2030:
                return True
        except:
            pass
        return False

    # Map check names to functions
    CHECK_FUNCTIONS = {
        'solid': _check_solid,
        'palindrome': _check_palindrome,
        'repeater': _check_repeater,
        'ladder_asc': _check_ladder_asc,
        'ladder_desc': _check_ladder_desc,
        'all_even': _check_all_even,
        'all_odd': _check_all_odd,
        'binary_digits': _check_binary_digits,
        'alternator': _check_alternator,
        'four_pairs': _check_four_pairs,
        'three_pairs': _check_three_pairs,
        'full_house': _check_full_house,
        'seven_of_kind': _check_seven_of_kind,
        'two_pair_triple': _check_two_pair_triple,
        'triple_double_double': _check_triple_double_double,
        'consecutive_triples': _check_consecutive_triples,
        'chunky_ladder': _check_chunky_ladder,
        'doubles_ladder': _check_doubles_ladder,
        'flipper_digits': _check_flipper_digits,
        'true_flipper': _check_true_flipper,
        'near_flipper': _check_near_flipper,
        'broken_radar': _check_broken_radar,
        'sequential_trinary': _check_sequential_trinary,
        'pyramid_ladder': _check_pyramid_ladder,
        'counting_ladder': _check_counting_ladder,
        'step_ladder': _check_step_ladder,
        'super_ladder': _check_super_ladder,
        'quad_symmetry': _check_quad_symmetry,
        'ladder_and_quad': _check_ladder_and_quad,
        'birthday': _check_birthday,
        'ladder_4': lambda self, d: self._check_ladder_n(d, 4),
        'ladder_5': lambda self, d: self._check_ladder_n(d, 5),
        'ladder_6': lambda self, d: self._check_ladder_n(d, 6),
        'ladder_7': lambda self, d: self._check_ladder_n(d, 7),
        'counting_2s': lambda self, d: self._check_counting_step(d, 2),
        'counting_3s': lambda self, d: self._check_counting_step(d, 3),
        'counting_4s': lambda self, d: self._check_counting_step(d, 4),
        'counting_5s': lambda self, d: self._check_counting_step(d, 5),
        'counting_6s': lambda self, d: self._check_counting_step(d, 6),
        'counting_7s': lambda self, d: self._check_counting_step(d, 7),
        'counting_8s': lambda self, d: self._check_counting_step(d, 8),
        'counting_9s': lambda self, d: self._check_counting_step(d, 9),
    }

    # =========================================================================
    # RULE EVALUATION
    # =========================================================================

    def _evaluate_rule(self, rule_type: str, rule_value, digits: str, full_serial: str, metadata: dict = None) -> bool:
        """Evaluate a single rule.

        Args:
            rule_type: Type of rule (regex, contains, baseline_variance_min, etc.)
            rule_value: Value to compare against
            digits: Numeric portion of serial
            full_serial: Complete serial string
            metadata: Optional dict with detection metadata (baseline_variance, etc.)
        """
        metadata = metadata or {}

        if rule_type == 'regex':
            return bool(re.search(rule_value, digits))

        elif rule_type == 'contains':
            return rule_value in digits

        elif rule_type == 'starts_with':
            return digits.startswith(rule_value)

        elif rule_type == 'ends_with':
            return digits.endswith(rule_value) or full_serial.endswith(rule_value)

        elif rule_type == 'unique_count':
            return len(set(digits)) == rule_value

        elif rule_type == 'unique_max':
            return len(set(digits)) <= rule_value

        elif rule_type == 'digit_sum':
            return sum(int(d) for d in digits) == rule_value

        elif rule_type == 'digit_sum_min':
            return sum(int(d) for d in digits) >= rule_value

        elif rule_type == 'digit_sum_max':
            return sum(int(d) for d in digits) <= rule_value

        elif rule_type == 'baseline_variance_min':
            # For gas pump detection - unusually tall bounding box
            baseline_variance = metadata.get('baseline_variance', 0.0)
            return baseline_variance >= rule_value

        elif rule_type == 'baseline_variance_max':
            baseline_variance = metadata.get('baseline_variance', 0.0)
            return baseline_variance <= rule_value

        elif rule_type == 'check':
            check_fn = self.CHECK_FUNCTIONS.get(rule_value)
            if check_fn:
                return check_fn(self, digits)
            return False

        elif rule_type == 'all':
            return all(
                self._evaluate_rules(sub_rule, digits, full_serial, metadata)
                for sub_rule in rule_value
            )

        elif rule_type == 'any':
            return any(
                self._evaluate_rules(sub_rule, digits, full_serial, metadata)
                for sub_rule in rule_value
            )

        return False

    def _evaluate_rules(self, rules: dict, digits: str, full_serial: str, metadata: dict = None) -> bool:
        """Evaluate a rules dict."""
        for rule_type, rule_value in rules.items():
            if not self._evaluate_rule(rule_type, rule_value, digits, full_serial, metadata):
                return False
        return True

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def extract_digits(self, serial: str) -> str:
        """Extract numeric portion of serial."""
        return ''.join(c for c in serial if c.isdigit())

    def classify(self, serial: str, metadata: dict = None) -> List[PatternMatch]:
        """
        Classify a serial number.
        Returns list of PatternMatch sorted by tier.

        Args:
            serial: The serial number to classify
            metadata: Optional dict with detection metadata (baseline_variance, etc.)
                      Used for printing error patterns like GAS_PUMP
        """
        if not serial:
            return []

        digits = self.extract_digits(serial)
        if len(digits) != 8:
            return []

        matches = []
        metadata = metadata or {}

        for name, defn in self.patterns.items():
            rules = defn.get('rules', {})
            uses_full = defn.get('uses_full_serial', False)

            try:
                if self._evaluate_rules(rules, digits, serial, metadata):
                    matches.append(PatternMatch(
                        name=name,
                        description=defn.get('description', ''),
                        tier=defn.get('tier', 10)
                    ))
            except Exception:
                continue

        matches.sort(key=lambda m: (m.tier, m.name))
        return matches

    def classify_simple(self, serial: str, metadata: dict = None) -> List[str]:
        """Return just pattern names."""
        return [m.name for m in self.classify(serial, metadata)]

    def get_pattern_info(self, name: str) -> Optional[dict]:
        """Get info about a pattern."""
        return self.patterns.get(name)

    def get_all_patterns(self) -> dict:
        """Get all patterns."""
        return self.patterns.copy()

    def get_patterns_by_tier(self, tier: int) -> dict:
        """Get patterns of a specific tier."""
        return {k: v for k, v in self.patterns.items() if v.get('tier') == tier}

    def set_pattern_enabled(self, name: str, enabled: bool):
        """Enable/disable a pattern (stored in user config, not main file)."""
        # Initialize lists if needed
        if 'disabled_patterns' not in self.user_config:
            self.user_config['disabled_patterns'] = []
        if 'enabled_patterns' not in self.user_config:
            self.user_config['enabled_patterns'] = []

        disabled = self.user_config['disabled_patterns']
        enabled_list = self.user_config['enabled_patterns']

        # Get the default state from main config
        main_patterns = self.config.get('patterns', {})
        default_enabled = True
        if name in main_patterns and main_patterns[name]:
            default_enabled = main_patterns[name].get('enabled', True)

        if enabled:
            # User wants it enabled
            if name in disabled:
                disabled.remove(name)
            # Only add to enabled if it's disabled by default
            if not default_enabled and name not in enabled_list:
                enabled_list.append(name)
        else:
            # User wants it disabled
            if name in enabled_list:
                enabled_list.remove(name)
            # Only add to disabled if it's enabled by default
            if default_enabled and name not in disabled:
                disabled.append(name)

        # Handle user custom patterns
        user_custom = self.user_config.get('custom_patterns', {})
        if name in user_custom:
            user_custom[name]['enabled'] = enabled

        self.patterns = self._build_patterns()

    def add_custom_pattern(self, name: str, defn: dict):
        """Add a custom pattern to user config."""
        if 'custom_patterns' not in self.user_config:
            self.user_config['custom_patterns'] = {}
        self.user_config['custom_patterns'][name] = defn
        self.patterns = self._build_patterns()

    def remove_custom_pattern(self, name: str):
        """Remove a custom pattern from user config."""
        if 'custom_patterns' in self.user_config and name in self.user_config['custom_patterns']:
            del self.user_config['custom_patterns'][name]
        self.patterns = self._build_patterns()

    def get_custom_patterns(self) -> dict:
        """Get all custom patterns from user config."""
        return self.user_config.get('custom_patterns', {}).copy()

    def get_gas_pump_threshold(self) -> float:
        """Get the GAS_PUMP baseline_variance_min threshold.

        Checks user overrides first, then main config, then defaults to 3.5.
        """
        # Check user overrides first
        overrides = self.user_config.get('pattern_overrides', {})
        if 'GAS_PUMP' in overrides and 'baseline_variance_min' in overrides['GAS_PUMP']:
            return float(overrides['GAS_PUMP']['baseline_variance_min'])

        # Check main config
        patterns = self.config.get('patterns', {})
        if 'GAS_PUMP' in patterns:
            rules = patterns['GAS_PUMP'].get('rules', {})
            if 'baseline_variance_min' in rules:
                return float(rules['baseline_variance_min'])

        # Default
        return 3.5

    def set_gas_pump_threshold(self, threshold: float):
        """Set the GAS_PUMP baseline_variance_min threshold and save.

        Updates the user config overrides and saves to file.
        """
        if 'pattern_overrides' not in self.user_config:
            self.user_config['pattern_overrides'] = {}
        if 'GAS_PUMP' not in self.user_config['pattern_overrides']:
            self.user_config['pattern_overrides']['GAS_PUMP'] = {}

        self.user_config['pattern_overrides']['GAS_PUMP']['baseline_variance_min'] = threshold
        self.save_config()

        # Rebuild patterns to pick up the new threshold
        self.patterns = self._build_patterns()

    def save_config(self):
        """Save user config to file (preserves main patterns file)."""
        with open(self.user_config_path, 'w') as f:
            yaml.dump(self.user_config, f, default_flow_style=False, sort_keys=False)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    engine = PatternEngine()

    test_serials = [
        ("A88888888B", "SOLID"),
        ("A12344321B", "RADAR"),
        ("A12341234B", "REPEATER"),
        ("A12121212B", "SUPER_REPEATER"),
        ("A01234567B", "LADDER"),
        ("A11211121B", "BINARY"),
        ("A00000001B", "LOW_SERIAL"),
        ("A12345678*", "STAR"),
        ("A12333345B", "QUADS"),
        ("A11223344B", "CHUNKY_LADDER"),
        ("A12777456B", "LUCKY_777"),
    ]

    print("Pattern Engine v2 - Complete Test")
    print("=" * 70)

    for serial, expected in test_serials:
        matches = engine.classify_simple(serial)
        status = "✓" if expected in matches else "✗"
        print(f"{status} {serial}: {', '.join(matches[:5])}")
        if len(matches) > 5:
            print(f"    ... and {len(matches)-5} more")

    print(f"\nTotal patterns loaded: {len(engine.patterns)}")
    print(f"Tiers: {sorted(set(p.get('tier', 10) for p in engine.patterns.values()))}")
