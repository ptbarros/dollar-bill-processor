"""
Pattern Engine v2 - Complete Implementation

Evaluates serial numbers against patterns defined in patterns_v2.yaml.
Single source of truth for CLI, GUI, and multi-denomination support.
"""

import re
import yaml
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING
from dataclasses import dataclass
from collections import Counter

if TYPE_CHECKING:
    from settings_manager import SettingsManager


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

    def __init__(self, config_path: Path = None, settings: 'SettingsManager' = None):
        if config_path is None:
            self.config_path = Path(__file__).parent / "patterns_v2.yaml"
        else:
            self.config_path = Path(config_path) if isinstance(config_path, str) else config_path
        self.user_config_path = self.config_path.parent / "user_patterns.yaml"

        # Use provided settings manager or get global instance
        self._settings = settings
        if self._settings is None:
            try:
                from settings_manager import get_settings
                self._settings = get_settings()
            except ImportError:
                self._settings = None

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
        """Load user-specific pattern settings from SettingsManager.

        Returns a dict in the legacy format for backward compatibility:
        - custom_patterns: dict of user-defined YAML patterns
        - disabled_patterns: list of pattern names to disable
        - enabled_patterns: list of pattern names to enable
        - pattern_overrides: dict of rule overrides per pattern
        """
        # If we have a settings manager, use it as the source of truth
        if self._settings is not None:
            # Build disabled/enabled lists from pattern_states
            disabled = []
            enabled = []
            for name, is_enabled in self._settings.pattern_states.items():
                if is_enabled:
                    enabled.append(name)
                else:
                    disabled.append(name)

            return {
                'custom_patterns': self._settings.custom_patterns.copy(),
                'disabled_patterns': disabled,
                'enabled_patterns': enabled,
                'pattern_overrides': self._settings.pattern_overrides.copy(),
            }

        # Fallback to reading from user_patterns.yaml (legacy)
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
        """Triple + two pairs with CONSECUTIVE runs.

        Must have digits grouped in a row, e.g., 11133224 (111+33+22+4).
        """
        if len(digits) != 8:
            return False
        # Get consecutive run lengths
        runs = []
        i = 0
        while i < len(digits):
            run_len = 1
            while i + run_len < len(digits) and digits[i + run_len] == digits[i]:
                run_len += 1
            runs.append(run_len)
            i += run_len
        # Sort runs to check pattern
        sorted_runs = sorted(runs, reverse=True)
        return sorted_runs in [[3, 2, 2, 1], [3, 2, 2]]

    def _check_triple_double_double(self, digits: str) -> bool:
        """Triple + double + double pattern with CONSECUTIVE runs.

        Must have digits grouped in a row, e.g., 11122334 (111+22+33+4).
        Scattered digits like 49956594 don't count.
        """
        if len(digits) != 8:
            return False
        # Get consecutive run lengths
        runs = []
        i = 0
        while i < len(digits):
            run_len = 1
            while i + run_len < len(digits) and digits[i + run_len] == digits[i]:
                run_len += 1
            runs.append(run_len)
            i += run_len
        # Sort runs to check pattern: need exactly one 3, two 2s, and one 1
        sorted_runs = sorted(runs, reverse=True)
        return sorted_runs == [3, 2, 2, 1]

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
        """All digits readable when flipped upside down (0,1,6,8,9 only).

        Unlike true_flipper which must read the SAME when flipped,
        near_flipper just needs to be READABLE when flipped.
        Rate: ~1 in 256 (0.4%)
        """
        if len(digits) != 8:
            return False
        return set(digits).issubset({'0', '1', '6', '8', '9'})

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
        """Enable/disable a pattern (stored in SettingsManager or user config)."""
        # Update SettingsManager if available
        if self._settings is not None:
            self._settings.set_pattern_enabled(name, enabled)

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
        """Add a custom pattern to SettingsManager or user config."""
        if self._settings is not None:
            self._settings.set_custom_pattern(name, defn)
            # Update local user_config for consistency
            if 'custom_patterns' not in self.user_config:
                self.user_config['custom_patterns'] = {}
            self.user_config['custom_patterns'][name] = defn
        else:
            if 'custom_patterns' not in self.user_config:
                self.user_config['custom_patterns'] = {}
            self.user_config['custom_patterns'][name] = defn
        self.patterns = self._build_patterns()

    def remove_custom_pattern(self, name: str):
        """Remove a custom pattern from SettingsManager or user config."""
        if self._settings is not None:
            self._settings.remove_custom_pattern(name)
            # Update local user_config for consistency
            if 'custom_patterns' in self.user_config and name in self.user_config['custom_patterns']:
                del self.user_config['custom_patterns'][name]
        else:
            if 'custom_patterns' in self.user_config and name in self.user_config['custom_patterns']:
                del self.user_config['custom_patterns'][name]
        self.patterns = self._build_patterns()

    def get_custom_patterns(self) -> dict:
        """Get all custom patterns from SettingsManager or user config."""
        if self._settings is not None:
            return self._settings.custom_patterns.copy()
        return self.user_config.get('custom_patterns', {}).copy()

    def get_gas_pump_threshold(self) -> float:
        """Get the GAS_PUMP baseline_variance_min threshold.

        Checks SettingsManager first, then main config, then defaults to 3.5.
        """
        # Check settings manager first
        if self._settings is not None:
            override = self._settings.get_pattern_override('GAS_PUMP', 'baseline_variance_min')
            if override is not None:
                return float(override)

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

        Updates the SettingsManager and saves to file.
        """
        if self._settings is not None:
            self._settings.set_gas_pump_threshold(threshold)
            self._settings.save()
            # Rebuild user_config to reflect the change
            self.user_config = self._load_user_config()
        else:
            # Legacy fallback
            if 'pattern_overrides' not in self.user_config:
                self.user_config['pattern_overrides'] = {}
            if 'GAS_PUMP' not in self.user_config['pattern_overrides']:
                self.user_config['pattern_overrides']['GAS_PUMP'] = {}
            self.user_config['pattern_overrides']['GAS_PUMP']['baseline_variance_min'] = threshold
            self.save_config()

        # Rebuild patterns to pick up the new threshold
        self.patterns = self._build_patterns()

    def save_config(self):
        """Save user config via SettingsManager (or legacy file).

        When SettingsManager is available, syncs custom_patterns and
        pattern_overrides to user_settings.yaml.

        NOTE: Pattern enabled/disabled states are NOT synced here anymore.
        The v3 engine manages pattern_states directly via set_pattern_enabled()
        and clear_pattern_enabled(). Syncing from user_config would overwrite
        those changes with stale data.
        """
        if self._settings is not None:
            # Sync custom patterns
            self._settings.custom_patterns = self.user_config.get('custom_patterns', {}).copy()

            # Sync pattern overrides
            for pattern_name, overrides in self.user_config.get('pattern_overrides', {}).items():
                if pattern_name not in self._settings.pattern_overrides:
                    self._settings.pattern_overrides[pattern_name] = {}
                self._settings.pattern_overrides[pattern_name].update(overrides)

            self._settings.save()
        else:
            # Legacy fallback
            with open(self.user_config_path, 'w') as f:
                yaml.dump(self.user_config, f, default_flow_style=False, sort_keys=False)

    def get_digit_highlights(self, serial: str, matched_patterns: List[str]) -> dict:
        """Get highlight and visualization info for pattern overlay.

        Returns a dict with:
        - 'highlights': list of highlight specs per digit position (0-7)
          Each: {position, digit, highlights: [{pattern, color, reason}]}
        - 'connectors': list of connector lines to draw between digit pairs
          Each: {positions: [i, j], color, pattern}

        Colors are CSS-style for easy UI use:
        - 'purple': flipper-related digits (0,1,6,8,9)
        - 'blue': binary digits (0,1)
        - 'cyan': trinary digits
        - 'orange': radar matching pairs
        - 'magenta': repeater pattern digits
        - 'yellow': solid/near-solid dominant digit
        - 'lime': ladder sequence digits
        """
        # Extract just digits (no prefix/suffix letters)
        digits = ''.join(c for c in serial if c.isdigit())
        if len(digits) != 8:
            return {'highlights': [], 'connectors': []}

        # Initialize highlight info for each position
        highlights = []
        for i, d in enumerate(digits):
            highlights.append({
                'position': i,
                'digit': d,
                'highlights': []
            })

        # Connectors for relational patterns (radar pairs, etc.)
        connectors = []

        # Define flipper-valid digits
        FLIPPER_DIGITS = {'0', '1', '6', '8', '9'}
        BINARY_DIGITS = {'0', '1'}

        for pattern in matched_patterns:
            pattern_upper = pattern.upper()

            # Flipper-related patterns: highlight flip-valid digits
            if pattern_upper in ('FLIPPER', 'TRUE_FLIPPER', 'NEAR_FLIPPER', 'GAS_PUMP_FLIPPER'):
                for i, d in enumerate(digits):
                    if d in FLIPPER_DIGITS:
                        highlights[i]['highlights'].append({
                            'pattern': pattern,
                            'color': 'purple',
                            'reason': 'flip-valid digit'
                        })

            # Binary: highlight 0s and 1s
            elif pattern_upper == 'BINARY':
                for i, d in enumerate(digits):
                    if d in BINARY_DIGITS:
                        highlights[i]['highlights'].append({
                            'pattern': pattern,
                            'color': 'blue',
                            'reason': 'binary digit'
                        })

            # Trinary: highlight all digits (they're all part of the 3 unique values)
            elif pattern_upper == 'TRINARY':
                unique_digits = set(digits)
                for i, d in enumerate(digits):
                    highlights[i]['highlights'].append({
                        'pattern': pattern,
                        'color': 'cyan',
                        'reason': f'one of {len(unique_digits)} unique digits'
                    })

            # Radar (palindrome): highlight matching pairs and add connectors
            elif pattern_upper == 'RADAR':
                # Colors for each pair (to distinguish them visually)
                pair_colors = ['orange', 'coral', 'gold', 'salmon']
                for i in range(4):
                    j = 7 - i  # mirror position
                    if digits[i] == digits[j]:
                        pair_color = pair_colors[i]
                        highlights[i]['highlights'].append({
                            'pattern': pattern,
                            'color': pair_color,
                            'reason': f'palindrome pair with pos {j+1}'
                        })
                        highlights[j]['highlights'].append({
                            'pattern': pattern,
                            'color': pair_color,
                            'reason': f'palindrome pair with pos {i+1}'
                        })
                        # Add connector line between the pair
                        connectors.append({
                            'positions': [i, j],
                            'color': pair_color,
                            'pattern': pattern
                        })

            # Broken Radar: one pair doesn't match
            elif pattern_upper == 'BROKEN_RADAR':
                pair_colors = ['orange', 'coral', 'gold', 'salmon']
                for i in range(4):
                    j = 7 - i  # mirror position
                    if digits[i] == digits[j]:
                        # Matching pair - show with connector
                        pair_color = pair_colors[i]
                        highlights[i]['highlights'].append({
                            'pattern': pattern,
                            'color': pair_color,
                            'reason': f'matching pair with pos {j+1}'
                        })
                        highlights[j]['highlights'].append({
                            'pattern': pattern,
                            'color': pair_color,
                            'reason': f'matching pair with pos {i+1}'
                        })
                        connectors.append({
                            'positions': [i, j],
                            'color': pair_color,
                            'pattern': pattern
                        })
                    else:
                        # Broken pair - highlight in red
                        highlights[i]['highlights'].append({
                            'pattern': pattern,
                            'color': 'red',
                            'reason': f'broken pair (should match pos {j+1})'
                        })
                        highlights[j]['highlights'].append({
                            'pattern': pattern,
                            'color': 'red',
                            'reason': f'broken pair (should match pos {i+1})'
                        })
                        # Add dashed-style connector for broken pair
                        connectors.append({
                            'positions': [i, j],
                            'color': 'red',
                            'pattern': pattern,
                            'style': 'broken'  # Signal to draw differently
                        })

            # Repeater (ABCDABCD): highlight the repeat
            elif pattern_upper == 'REPEATER':
                if digits[:4] == digits[4:]:
                    for i in range(8):
                        highlights[i]['highlights'].append({
                            'pattern': pattern,
                            'color': 'magenta',
                            'reason': 'repeating group'
                        })

            # Solid: highlight all (same digit)
            elif pattern_upper == 'SOLID':
                for i in range(8):
                    highlights[i]['highlights'].append({
                        'pattern': pattern,
                        'color': 'yellow',
                        'reason': 'solid digit'
                    })

            # Near solid: highlight the dominant digit
            elif pattern_upper == 'NEAR_SOLID':
                counter = Counter(digits)
                dominant = counter.most_common(1)[0][0]
                for i, d in enumerate(digits):
                    if d == dominant:
                        highlights[i]['highlights'].append({
                            'pattern': pattern,
                            'color': 'yellow',
                            'reason': 'dominant digit'
                        })

            # Ladder patterns: highlight all digits in sequence
            elif 'LADDER' in pattern_upper:
                for i in range(8):
                    highlights[i]['highlights'].append({
                        'pattern': pattern,
                        'color': 'lime',
                        'reason': 'ladder sequence'
                    })

            # Quads/Trips/etc: highlight runs of same digit
            elif pattern_upper in ('QUADS', 'QUINTS', 'SIXES', 'SEVENS', 'TRIPS', 'DOUBLE_QUADS'):
                # Find runs of same digit
                i = 0
                while i < len(digits):
                    run_digit = digits[i]
                    run_start = i
                    run_len = 1
                    while i + run_len < len(digits) and digits[i + run_len] == run_digit:
                        run_len += 1
                    # Highlight runs of 3+
                    if run_len >= 3:
                        for j in range(run_start, run_start + run_len):
                            highlights[j]['highlights'].append({
                                'pattern': pattern,
                                'color': 'gold',
                                'reason': f'run of {run_len}'
                            })
                    i += run_len

            # Pairs patterns: highlight pairs
            elif 'PAIR' in pattern_upper or pattern_upper in ('FOUR_PAIRS', 'THREE_PAIRS'):
                # Find consecutive pairs
                i = 0
                while i < len(digits) - 1:
                    if digits[i] == digits[i + 1]:
                        highlights[i]['highlights'].append({
                            'pattern': pattern,
                            'color': 'teal',
                            'reason': 'pair'
                        })
                        highlights[i + 1]['highlights'].append({
                            'pattern': pattern,
                            'color': 'teal',
                            'reason': 'pair'
                        })
                        i += 2
                    else:
                        i += 1

        return {'highlights': highlights, 'connectors': connectors}


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
