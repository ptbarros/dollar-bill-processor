#!/usr/bin/env python3
"""
Dollar Bill Production Pipeline - Phase 1
Automated processing of scanned dollar bills with fancy number detection and cropping.

Features:
- Scanner-agnostic input (auto-detects naming conventions)
- Front/back detection using YOLO serial count
- Smart pairing of front/back images
- Percentage-based cropping (scanner-independent)
- Fancy number filtering (only crops fancy bills)
- Stack position tracking
- Batch cleanup helper

Usage:
    python process_production.py /path/to/scans --output fancy_bills/
    python process_production.py /path/to/scans --output fancy_bills/ --reference /path/to/ref.jpg
"""

import cv2
import numpy as np
import easyocr
import re
from pathlib import Path
from ultralytics import YOLO
import csv
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import argparse
import shutil
import yaml


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================

class Config:
    """Loads and manages configuration from config.yaml and patterns.txt"""

    # Default values (used if config.yaml is missing)
    DEFAULT_CROP_REGIONS = {
        'front': {
            'full':   {'x': 0.0,   'y': 0.0,   'w': 1.0,   'h': 1.0},
            'seal':   {'x': 0.586, 'y': 0.183, 'w': 0.254, 'h': 0.537},
            'left':   {'x': 0.0,   'y': 0.0,   'w': 0.384, 'h': 1.0},
            'center': {'x': 0.354, 'y': 0.0,   'w': 0.322, 'h': 1.0},
            'right':  {'x': 0.627, 'y': 0.0,   'w': 0.373, 'h': 1.0},
        },
        'back': {
            'full':   {'x': 0.0,   'y': 0.0,   'w': 1.0,   'h': 1.0},
            'seal':   {'x': 0.619, 'y': 0.221, 'w': 0.261, 'h': 0.557},
            'left':   {'x': 0.0,   'y': 0.0,   'w': 0.384, 'h': 1.0},
            'center': {'x': 0.354, 'y': 0.0,   'w': 0.322, 'h': 1.0},
            'right':  {'x': 0.627, 'y': 0.0,   'w': 0.373, 'h': 1.0},
        }
    }

    DEFAULT_CROP_ORDER = [
        ['front', 'seal'], ['front', 'full'], ['front', 'left'],
        ['front', 'center'], ['front', 'right'], ['back', 'seal'],
        ['back', 'full'], ['back', 'left'], ['back', 'center'], ['back', 'right']
    ]

    def __init__(self, config_path: Optional[Path] = None, patterns_path: Optional[Path] = None):
        self.config_path = config_path
        self.patterns_path = patterns_path
        self.data = {}
        self._patterns_from_txt = None

        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                self.data = yaml.safe_load(f) or {}
            print(f"Loaded config: {config_path}")
        else:
            print("Using default configuration")

        # Load patterns from simple text file if it exists
        if patterns_path and patterns_path.exists():
            self._patterns_from_txt = self._load_patterns_txt(patterns_path)
            print(f"Loaded patterns: {patterns_path} ({len(self._patterns_from_txt)} patterns)")

    def _load_patterns_txt(self, path: Path) -> list:
        """Load patterns from simple text file format: NAME,TYPE,VALUE"""
        patterns = []
        with open(path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                parts = line.split(',', 2)  # Split into max 3 parts
                if len(parts) < 3:
                    print(f"  Warning: Skipping invalid pattern on line {line_num}: {line}")
                    continue

                name, pattern_type, value = parts[0].strip(), parts[1].strip(), parts[2].strip()
                patterns.append({
                    'name': name,
                    'type': pattern_type,
                    'value': value
                })
        return patterns

    @property
    def crop_regions(self) -> dict:
        """Get crop regions from config or defaults."""
        if 'crops' in self.data:
            regions = {}
            for side in ['front', 'back']:
                if side in self.data['crops']:
                    regions[side] = {}
                    for name, coords in self.data['crops'][side].items():
                        regions[side][name] = {
                            'x': coords.get('x', 0),
                            'y': coords.get('y', 0),
                            'w': coords.get('w', 1),
                            'h': coords.get('h', 1)
                        }
            return regions
        return self.DEFAULT_CROP_REGIONS

    @property
    def crop_order(self) -> list:
        """Get crop order from config or defaults."""
        if 'crop_order' in self.data:
            return [tuple(item) for item in self.data['crop_order']]
        return [tuple(item) for item in self.DEFAULT_CROP_ORDER]

    @property
    def builtin_patterns(self) -> dict:
        """Get enabled/disabled status of built-in patterns."""
        defaults = {
            # Original core patterns
            'solid': True, 'repeater': True, 'radar': True,
            'ladder': True, 'low_serial': True, 'binary': True, 'star_note': True,
            # Digit count patterns
            'trinary': True, 'true_trinary': True, 'quadrinary': False,
            'true_quadrinary': True, 'quinary': False,
            # Even/odd patterns
            'all_evens': True, 'all_odds': True,
            # Special patterns
            'binary_radar': True, 'alternator': True,
            # Partial ladders
            'ladder_7': True, 'ladder_6': True, 'ladder_5': True, 'ladder_4': True,
            # Structural patterns
            'full_house': True, 'two_pair_triple': True, 'triple_double_double': True,
            'consecutive_triples': True, 'double_quads_sequential': True,
            # Ladder variants
            'pyramid_ladder': True, 'counting_ladder': True, 'step_ladder': False,
            'chunky_ladder': True, 'super_ladder': True,
            # Counting ladders (step patterns)
            'counting_2s': True, 'counting_3s': True, 'counting_4s': True,
            'counting_5s': True, 'counting_6s': True, 'counting_7s': True,
            'counting_8s': True, 'counting_9s': True,
            # Sum patterns
            'magic_sum': True,
            # Flipper patterns
            'flipper': True, 'true_flipper': True, 'near_flipper': False,
            # Special structural
            'broken_radar': True, 'sequential_trinary': True, 'double_bookend': True,
            'radar_repeater': True, 'birthday': False,
            # Additional patterns
            'four_pairs': True, 'three_pairs': True, 'ultra_low_serial': True,
            'super_radar': True, 'binary_repeater': True, 'doubles_ladder': True,
            'ladder_and_quad': True,
        }
        if 'patterns' in self.data and 'builtin' in self.data['patterns']:
            defaults.update(self.data['patterns']['builtin'])
        return defaults

    @property
    def custom_patterns(self) -> list:
        """Get list of custom patterns. Prefers patterns.txt over config.yaml."""
        # Prefer patterns.txt (simpler format, less error-prone)
        if self._patterns_from_txt is not None:
            return self._patterns_from_txt
        # Fall back to config.yaml
        if 'patterns' in self.data and 'custom' in self.data['patterns']:
            return self.data['patterns']['custom'] or []
        return []

    @property
    def jpeg_quality(self) -> int:
        """Get JPEG quality setting."""
        if 'options' in self.data:
            return self.data['options'].get('jpeg_quality', 95)
        return 95

    @property
    def star_notes_always_fancy(self) -> bool:
        """Whether star notes are always considered fancy."""
        if 'options' in self.data:
            return self.data['options'].get('star_notes_always_fancy', True)
        return True


# Global config instance (set during main())
config: Optional[Config] = None


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BillPair:
    """Represents a front/back pair of bill scans."""
    front_path: Path
    back_path: Optional[Path]
    stack_position: int  # Physical position in scanned stack
    serial: Optional[str] = None
    fancy_types: list = field(default_factory=list)
    confidence: float = 0.0
    is_fancy: bool = False
    error: Optional[str] = None


# =============================================================================
# CLASSES FROM process_bills_yolo.py (reused)
# =============================================================================

class BillAligner:
    """Aligns scanned bills to a reference template."""

    def __init__(self, reference_path):
        self.reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        if self.reference is None:
            raise ValueError(f"Could not load reference image: {reference_path}")

        self.orb = cv2.ORB_create(1000)
        self.ref_kp, self.ref_desc = self.orb.detectAndCompute(self.reference, None)

    def align_image(self, image_path):
        """Align a bill image to the reference. Returns color image."""
        img_color = cv2.imread(str(image_path))
        if img_color is None:
            return None

        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orb.detectAndCompute(img_gray, None)

        if desc is None or len(kp) < 10:
            return img_color

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(self.ref_desc, desc, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 10:
            return img_color

        ref_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in good_matches])
        img_pts = np.float32([kp[m.trainIdx].pt for m in good_matches])

        M, inliers = cv2.estimateAffinePartial2D(img_pts, ref_pts, method=cv2.RANSAC)

        if M is None:
            return img_color

        aligned = cv2.warpAffine(
            img_color, M, (self.reference.shape[1], self.reference.shape[0]),
            flags=cv2.INTER_CUBIC
        )
        return aligned


class FancyNumberDetector:
    """Detects fancy serial numbers (repeaters, radars, etc.)"""

    def __init__(self, cfg: Optional[Config] = None):
        """Initialize with optional config for custom patterns."""
        self.cfg = cfg

    @staticmethod
    def extract_digits(serial):
        """Extract just the 8 digits from serial (remove letters)"""
        if len(serial) >= 10:
            return serial[1:9]
        return None

    @staticmethod
    def is_repeater(digits):
        """Check if first 4 digits repeat (e.g., 12341234)"""
        if len(digits) != 8:
            return False
        return digits[:4] == digits[4:]

    @staticmethod
    def is_radar(digits):
        """Check if it's a palindrome (e.g., 12344321)"""
        if len(digits) != 8:
            return False
        return digits == digits[::-1]

    @staticmethod
    def is_solid(digits):
        """Check if all digits are the same (e.g., 11111111)"""
        if len(digits) != 8:
            return False
        return len(set(digits)) == 1

    @staticmethod
    def is_ladder(digits):
        """Check if digits are sequential (e.g., 12345678 or 87654321)"""
        if len(digits) != 8:
            return False
        nums = [int(d) for d in digits]
        if all(nums[i] + 1 == nums[i + 1] for i in range(7)):
            return True
        if all(nums[i] - 1 == nums[i + 1] for i in range(7)):
            return True
        return False

    @staticmethod
    def is_low_serial(digits):
        """Check if serial number is very low (< 100)"""
        if len(digits) != 8:
            return False
        try:
            return int(digits) <= 100
        except:
            return False

    @staticmethod
    def is_binary(digits):
        """Check if only contains 0s and 1s"""
        if len(digits) != 8:
            return False
        return set(digits).issubset({'0', '1'})

    @staticmethod
    def is_star_note(serial):
        """Check if it's a star note (ends with *)"""
        return serial and serial.endswith('*')

    # =========================================================================
    # DIGIT COUNT PATTERNS
    # =========================================================================

    @staticmethod
    def is_trinary(digits):
        """Check if uses only 3 or fewer different digits."""
        return len(digits) == 8 and len(set(digits)) <= 3

    @staticmethod
    def is_true_trinary(digits):
        """Check if uses exactly 3 different digits (each appears at least once)."""
        return len(digits) == 8 and len(set(digits)) == 3

    @staticmethod
    def is_quadrinary(digits):
        """Check if uses only 4 or fewer different digits."""
        return len(digits) == 8 and len(set(digits)) <= 4

    @staticmethod
    def is_true_quadrinary(digits):
        """Check if uses exactly 4 different digits."""
        return len(digits) == 8 and len(set(digits)) == 4

    @staticmethod
    def is_quinary(digits):
        """Check if uses only 5 or fewer different digits."""
        return len(digits) == 8 and len(set(digits)) <= 5

    # =========================================================================
    # EVEN/ODD PATTERNS
    # =========================================================================

    @staticmethod
    def is_all_evens(digits):
        """Check if all digits are even (0,2,4,6,8)."""
        return len(digits) == 8 and all(d in '02468' for d in digits)

    @staticmethod
    def is_all_odds(digits):
        """Check if all digits are odd (1,3,5,7,9)."""
        return len(digits) == 8 and all(d in '13579' for d in digits)

    # =========================================================================
    # SPECIAL PATTERNS
    # =========================================================================

    @staticmethod
    def has_ladder_of_length(digits, length):
        """Check if contains an ascending or descending ladder of given length."""
        if len(digits) < length:
            return False
        nums = [int(d) for d in digits]
        for i in range(len(nums) - length + 1):
            segment = nums[i:i+length]
            # Ascending
            if all(segment[j] + 1 == segment[j+1] for j in range(length-1)):
                return True
            # Descending
            if all(segment[j] - 1 == segment[j+1] for j in range(length-1)):
                return True
        return False

    @staticmethod
    def is_binary_radar(digits):
        """Check if binary AND radar (palindrome with only 0s and 1s)."""
        if len(digits) != 8:
            return False
        return set(digits).issubset({'0', '1'}) and digits == digits[::-1]

    @staticmethod
    def is_alternator(digits):
        """Check if digits alternate between two values (e.g., 12121212)."""
        if len(digits) != 8:
            return False
        if len(set(digits)) != 2:
            return False
        return all(digits[i] == digits[i % 2] for i in range(8))

    # =========================================================================
    # STRUCTURAL PATTERNS
    # =========================================================================

    @staticmethod
    def is_full_house(digits):
        """Check for full house: 5 of one digit, 3 of another."""
        if len(digits) != 8:
            return False
        from collections import Counter
        counts = sorted(Counter(digits).values(), reverse=True)
        return counts == [5, 3]

    @staticmethod
    def is_two_pair_triple(digits):
        """Check for triple + two pairs pattern."""
        if len(digits) != 8:
            return False
        from collections import Counter
        counts = sorted(Counter(digits).values(), reverse=True)
        return counts == [3, 2, 2, 1] or counts == [3, 2, 2]

    @staticmethod
    def is_triple_double_double(digits):
        """Check for triple + double + double + single pattern."""
        if len(digits) != 8:
            return False
        from collections import Counter
        counts = sorted(Counter(digits).values(), reverse=True)
        return counts == [3, 2, 2, 1]

    @staticmethod
    def is_consecutive_triples(digits):
        """Check for two triples back-to-back (111222xx)."""
        if len(digits) != 8:
            return False
        # Check positions 0-5 for two consecutive triples
        for i in range(3):  # Start positions 0, 1, 2
            if (digits[i] == digits[i+1] == digits[i+2] and
                digits[i+3] == digits[i+4] == digits[i+5] and
                digits[i] != digits[i+3]):
                return True
        return False

    @staticmethod
    def is_double_quads_sequential(digits):
        """Check for two sequential quads (11112222)."""
        if len(digits) != 8:
            return False
        return (digits[0] == digits[1] == digits[2] == digits[3] and
                digits[4] == digits[5] == digits[6] == digits[7] and
                digits[0] != digits[4])

    # =========================================================================
    # LADDER VARIANTS
    # =========================================================================

    @staticmethod
    def is_pyramid_ladder(digits):
        """Check for pyramid pattern (up then down): 12321xxx."""
        if len(digits) < 5:
            return False
        nums = [int(d) for d in digits]
        # Check for 5-digit pyramids
        for i in range(len(nums) - 4):
            seg = nums[i:i+5]
            if (seg[0] < seg[1] < seg[2] and seg[2] > seg[3] > seg[4] and
                seg[0] == seg[4] and seg[1] == seg[3]):
                return True
        return False

    @staticmethod
    def is_counting_ladder(digits):
        """Check for counting pattern: 12123123 or similar."""
        if len(digits) != 8:
            return False
        # Pattern: 12123123
        if digits == '12123123':
            return True
        # Check for other counting patterns
        if digits[:2] == '12' and digits[2:5] == '123' and digits[5:] == '123':
            return True
        return False

    @staticmethod
    def is_step_ladder(digits):
        """Check for steps of 2: 02468xxx."""
        if len(digits) != 8:
            return False
        nums = [int(d) for d in digits]
        # Check for 4+ digit step pattern
        for i in range(5):
            if all(nums[i+j+1] - nums[i+j] == 2 for j in range(3)):
                return True
            if all(nums[i+j] - nums[i+j+1] == 2 for j in range(3)):
                return True
        return False

    @staticmethod
    def is_chunky_ladder(digits):
        """Check for paired ladder: 11223344."""
        if len(digits) != 8:
            return False
        # Check for ascending pairs
        if all(digits[i*2] == digits[i*2+1] for i in range(4)):
            nums = [int(digits[i*2]) for i in range(4)]
            if all(nums[i] + 1 == nums[i+1] for i in range(3)):
                return True
            if all(nums[i] - 1 == nums[i+1] for i in range(3)):
                return True
        return False

    @staticmethod
    def is_super_ladder(digits):
        """Check for double-digit ladder: 01020304."""
        if len(digits) != 8:
            return False
        # Pattern like 01020304, 02040608, etc.
        pairs = [digits[i:i+2] for i in range(0, 8, 2)]
        try:
            nums = [int(p) for p in pairs]
            diff = nums[1] - nums[0]
            if diff != 0 and all(nums[i+1] - nums[i] == diff for i in range(3)):
                return True
        except:
            pass
        return False

    # =========================================================================
    # SUM PATTERNS
    # =========================================================================

    @staticmethod
    def digit_sum(digits):
        """Calculate sum of all digits."""
        return sum(int(d) for d in digits)

    @staticmethod
    def is_magic_sum(digits):
        """Check if digit sum is a 'magic' number (9, 36, 72)."""
        if len(digits) != 8:
            return False
        s = sum(int(d) for d in digits)
        return s in {9, 36, 72}

    @staticmethod
    def get_sum_category(digits):
        """Return sum category if notable - collectors value extreme sums."""
        if len(digits) != 8:
            return None
        s = sum(int(d) for d in digits)
        # Collectors value: very low (1-11) and very high (61-72) sums
        # Min possible: 0 (00000000), Max possible: 72 (99999999)
        if s <= 11 or s >= 61:
            return f"SUM_{s}"
        return None

    @staticmethod
    def has_counting_ladder(digits, step):
        """Check for counting ladder with given step (e.g., step=2: 02040608)."""
        if len(digits) != 8:
            return False
        # Check as 4 two-digit pairs
        pairs = [digits[i:i+2] for i in range(0, 8, 2)]
        try:
            nums = [int(p) for p in pairs]
            # Check if each pair increases by step
            if all(nums[i+1] - nums[i] == step for i in range(3)):
                return True
            # Check descending
            if all(nums[i] - nums[i+1] == step for i in range(3)):
                return True
        except:
            pass
        return False

    # =========================================================================
    # FLIPPER PATTERNS (digits that look same upside down)
    # =========================================================================

    @staticmethod
    def is_flipper(digits):
        """Check if contains only flippable digits (0,1,6,8,9)."""
        return len(digits) == 8 and set(digits).issubset({'0', '1', '6', '8', '9'})

    @staticmethod
    def is_true_flipper(digits):
        """Check if reads same when flipped upside down."""
        if len(digits) != 8:
            return False
        flip_map = {'0': '0', '1': '1', '6': '9', '8': '8', '9': '6'}
        if not set(digits).issubset(set(flip_map.keys())):
            return False
        flipped = ''.join(flip_map[d] for d in reversed(digits))
        return digits == flipped

    @staticmethod
    def is_near_flipper(digits):
        """Check if one digit away from true flipper."""
        if len(digits) != 8:
            return False
        flip_map = {'0': '0', '1': '1', '6': '9', '8': '8', '9': '6'}
        non_flip_count = sum(1 for d in digits if d not in flip_map)
        return non_flip_count == 1

    # =========================================================================
    # SPECIAL STRUCTURAL PATTERNS
    # =========================================================================

    @staticmethod
    def is_broken_radar(digits):
        """Check if one digit away from being a radar."""
        if len(digits) != 8:
            return False
        reversed_digits = digits[::-1]
        differences = sum(1 for i in range(8) if digits[i] != reversed_digits[i])
        return differences == 2  # One swap = 2 position differences

    @staticmethod
    def is_sequential_trinary(digits):
        """Check for trinary with sequential digits (e.g., 121212 uses 1,2)."""
        if len(digits) != 8:
            return False
        unique = sorted(set(digits))
        if len(unique) not in [2, 3]:
            return False
        nums = [int(d) for d in unique]
        return all(nums[i] + 1 == nums[i+1] for i in range(len(nums)-1))

    @staticmethod
    def is_double_bookend(digits):
        """Check if first 2 digits equal last 2 digits."""
        return len(digits) == 8 and digits[:2] == digits[-2:]

    @staticmethod
    def is_radar_repeater(digits):
        """Check if both radar AND repeater."""
        if len(digits) != 8:
            return False
        is_radar = digits == digits[::-1]
        is_repeater = digits[:4] == digits[4:]
        return is_radar and is_repeater

    @staticmethod
    def is_birthday(digits):
        """Check if could be a date pattern (MMDDYYYY or DDMMYYYY)."""
        if len(digits) != 8:
            return False
        # MMDDYYYY
        try:
            mm, dd = int(digits[:2]), int(digits[2:4])
            yyyy = int(digits[4:])
            if 1 <= mm <= 12 and 1 <= dd <= 31 and 1900 <= yyyy <= 2030:
                return True
        except:
            pass
        # DDMMYYYY
        try:
            dd, mm = int(digits[:2]), int(digits[2:4])
            yyyy = int(digits[4:])
            if 1 <= mm <= 12 and 1 <= dd <= 31 and 1900 <= yyyy <= 2030:
                return True
        except:
            pass
        return False

    # =========================================================================
    # ADDITIONAL PATTERNS
    # =========================================================================

    @staticmethod
    def is_four_pairs(digits):
        """Check for four pairs: 11223344."""
        if len(digits) != 8:
            return False
        return all(digits[i*2] == digits[i*2+1] for i in range(4))

    @staticmethod
    def is_three_pairs(digits):
        """Check if contains at least three pairs."""
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

    @staticmethod
    def is_ultra_low_serial(digits):
        """Check if first 5+ digits are zeros."""
        if len(digits) != 8:
            return False
        return digits[:5] == '00000'

    @staticmethod
    def is_super_radar(digits):
        """Check for triple-symmetric radar (e.g., 12122121)."""
        if len(digits) != 8:
            return False
        # Must be radar first
        if digits != digits[::-1]:
            return False
        # Check for additional internal symmetry
        return digits[:4] == digits[:4][::-1]

    @staticmethod
    def is_binary_repeater(digits):
        """Check if binary AND repeater."""
        if len(digits) != 8:
            return False
        return set(digits).issubset({'0', '1'}) and digits[:4] == digits[4:]

    @staticmethod
    def is_doubles_ladder(digits):
        """Check for pairs in sequence: 11223344."""
        if len(digits) != 8:
            return False
        if not all(digits[i*2] == digits[i*2+1] for i in range(4)):
            return False
        nums = [int(digits[i*2]) for i in range(4)]
        # Check ascending or descending
        return (all(nums[i] + 1 == nums[i+1] for i in range(3)) or
                all(nums[i] - 1 == nums[i+1] for i in range(3)))

    @staticmethod
    def is_ladder_and_quad(digits):
        """Check if contains both a 4+ ladder and a quad."""
        if len(digits) != 8:
            return False
        # Check for quad
        has_quad = bool(re.search(r'(\d)\1{3}', digits))
        if not has_quad:
            return False
        # Check for 4-digit ladder
        nums = [int(d) for d in digits]
        for i in range(5):
            seg = nums[i:i+4]
            if (all(seg[j] + 1 == seg[j+1] for j in range(3)) or
                all(seg[j] - 1 == seg[j+1] for j in range(3))):
                return True
        return False

    def check_custom_pattern(self, pattern: dict, digits: str, serial: str) -> bool:
        """Check if digits/serial matches a custom pattern."""
        pattern_type = pattern.get('type', 'contains')
        value = pattern.get('value', '')

        if not value:
            return False

        if pattern_type == 'contains':
            return value in digits
        elif pattern_type == 'starts_with':
            return digits.startswith(value)
        elif pattern_type == 'ends_with':
            return digits.endswith(value)
        elif pattern_type == 'exact':
            return digits == value
        elif pattern_type == 'regex':
            try:
                return bool(re.search(value, digits))
            except re.error:
                return False
        elif pattern_type == 'serial_contains':
            # Match against full serial including letters
            return value in serial
        elif pattern_type == 'serial_regex':
            try:
                return bool(re.search(value, serial))
            except re.error:
                return False

        return False

    def classify(self, serial: str) -> list:
        """Classify a serial number and return all fancy types found."""
        if not serial:
            return []

        fancy_types = []

        # Get config settings
        builtin = self.cfg.builtin_patterns if self.cfg else {
            'solid': True, 'repeater': True, 'radar': True,
            'ladder': True, 'low_serial': True, 'binary': True, 'star_note': True
        }

        # Check star note first (uses full serial)
        if builtin.get('star_note', True) and self.is_star_note(serial):
            fancy_types.append("STAR")

        digits = self.extract_digits(serial)
        if not digits:
            return fancy_types

        # === Original core patterns ===
        if builtin.get('solid', True) and self.is_solid(digits):
            fancy_types.append("SOLID")
        if builtin.get('repeater', True) and self.is_repeater(digits):
            fancy_types.append("REPEATER")
        if builtin.get('radar', True) and self.is_radar(digits):
            fancy_types.append("RADAR")
        if builtin.get('ladder', True) and self.is_ladder(digits):
            fancy_types.append("LADDER")
        if builtin.get('low_serial', True) and self.is_low_serial(digits):
            fancy_types.append("LOW_SERIAL")
        if builtin.get('binary', True) and self.is_binary(digits):
            fancy_types.append("BINARY")

        # === Digit count patterns ===
        if builtin.get('trinary', True) and self.is_trinary(digits):
            fancy_types.append("TRINARY")
        if builtin.get('true_trinary', True) and self.is_true_trinary(digits):
            fancy_types.append("TRUE_TRINARY")
        if builtin.get('quadrinary', False) and self.is_quadrinary(digits):
            fancy_types.append("QUADRINARY")
        if builtin.get('true_quadrinary', True) and self.is_true_quadrinary(digits):
            fancy_types.append("TRUE_QUADRINARY")
        if builtin.get('quinary', False) and self.is_quinary(digits):
            fancy_types.append("QUINARY")

        # === Even/odd patterns ===
        if builtin.get('all_evens', True) and self.is_all_evens(digits):
            fancy_types.append("ALL_EVENS")
        if builtin.get('all_odds', True) and self.is_all_odds(digits):
            fancy_types.append("ALL_ODDS")

        # === Special patterns ===
        if builtin.get('binary_radar', True) and self.is_binary_radar(digits):
            fancy_types.append("BINARY_RADAR")
        if builtin.get('alternator', True) and self.is_alternator(digits):
            fancy_types.append("ALTERNATOR")

        # === Partial ladder patterns ===
        if builtin.get('ladder_7', True) and self.has_ladder_of_length(digits, 7):
            fancy_types.append("LADDER_7")
        if builtin.get('ladder_6', True) and self.has_ladder_of_length(digits, 6):
            fancy_types.append("LADDER_6")
        if builtin.get('ladder_5', True) and self.has_ladder_of_length(digits, 5):
            fancy_types.append("LADDER_5")
        if builtin.get('ladder_4', True) and self.has_ladder_of_length(digits, 4):
            fancy_types.append("LADDER_4")

        # === Structural patterns ===
        if builtin.get('full_house', True) and self.is_full_house(digits):
            fancy_types.append("FULL_HOUSE")
        if builtin.get('two_pair_triple', True) and self.is_two_pair_triple(digits):
            fancy_types.append("TWO_PAIR_TRIPLE")
        if builtin.get('triple_double_double', True) and self.is_triple_double_double(digits):
            fancy_types.append("TRIPLE_DOUBLE_DOUBLE")
        if builtin.get('consecutive_triples', True) and self.is_consecutive_triples(digits):
            fancy_types.append("CONSECUTIVE_TRIPLES")
        if builtin.get('double_quads_sequential', True) and self.is_double_quads_sequential(digits):
            fancy_types.append("DOUBLE_QUADS_SEQUENTIAL")

        # === Ladder variants ===
        if builtin.get('pyramid_ladder', True) and self.is_pyramid_ladder(digits):
            fancy_types.append("PYRAMID_LADDER")
        if builtin.get('counting_ladder', True) and self.is_counting_ladder(digits):
            fancy_types.append("COUNTING_LADDER")
        if builtin.get('step_ladder', False) and self.is_step_ladder(digits):
            fancy_types.append("STEP_LADDER")
        if builtin.get('chunky_ladder', True) and self.is_chunky_ladder(digits):
            fancy_types.append("CHUNKY_LADDER")
        if builtin.get('super_ladder', True) and self.is_super_ladder(digits):
            fancy_types.append("SUPER_LADDER")

        # === Sum patterns ===
        if builtin.get('magic_sum', True) and self.is_magic_sum(digits):
            fancy_types.append("MAGIC_SUM")
        sum_cat = self.get_sum_category(digits)
        if sum_cat:
            fancy_types.append(sum_cat)

        # === Counting ladder patterns (step of 2-9) ===
        if builtin.get('counting_2s', True) and self.has_counting_ladder(digits, 2):
            fancy_types.append("COUNTING_2S")
        if builtin.get('counting_3s', True) and self.has_counting_ladder(digits, 3):
            fancy_types.append("COUNTING_3S")
        if builtin.get('counting_4s', True) and self.has_counting_ladder(digits, 4):
            fancy_types.append("COUNTING_4S")
        if builtin.get('counting_5s', True) and self.has_counting_ladder(digits, 5):
            fancy_types.append("COUNTING_5S")
        if builtin.get('counting_6s', True) and self.has_counting_ladder(digits, 6):
            fancy_types.append("COUNTING_6S")
        if builtin.get('counting_7s', True) and self.has_counting_ladder(digits, 7):
            fancy_types.append("COUNTING_7S")
        if builtin.get('counting_8s', True) and self.has_counting_ladder(digits, 8):
            fancy_types.append("COUNTING_8S")
        if builtin.get('counting_9s', True) and self.has_counting_ladder(digits, 9):
            fancy_types.append("COUNTING_9S")

        # === Flipper patterns ===
        if builtin.get('flipper', True) and self.is_flipper(digits):
            fancy_types.append("FLIPPER")
        if builtin.get('true_flipper', True) and self.is_true_flipper(digits):
            fancy_types.append("TRUE_FLIPPER")
        if builtin.get('near_flipper', False) and self.is_near_flipper(digits):
            fancy_types.append("NEAR_FLIPPER")

        # === Special structural patterns ===
        if builtin.get('broken_radar', True) and self.is_broken_radar(digits):
            fancy_types.append("BROKEN_RADAR")
        if builtin.get('sequential_trinary', True) and self.is_sequential_trinary(digits):
            fancy_types.append("SEQUENTIAL_TRINARY")
        if builtin.get('double_bookend', True) and self.is_double_bookend(digits):
            fancy_types.append("DOUBLE_BOOKEND")
        if builtin.get('radar_repeater', True) and self.is_radar_repeater(digits):
            fancy_types.append("RADAR_REPEATER")
        if builtin.get('birthday', False) and self.is_birthday(digits):
            fancy_types.append("BIRTHDAY")

        # === Additional patterns ===
        if builtin.get('four_pairs', True) and self.is_four_pairs(digits):
            fancy_types.append("FOUR_PAIRS")
        if builtin.get('three_pairs', True) and self.is_three_pairs(digits):
            fancy_types.append("THREE_PAIRS")
        if builtin.get('ultra_low_serial', True) and self.is_ultra_low_serial(digits):
            fancy_types.append("ULTRA_LOW_SERIAL")
        if builtin.get('super_radar', True) and self.is_super_radar(digits):
            fancy_types.append("SUPER_RADAR")
        if builtin.get('binary_repeater', True) and self.is_binary_repeater(digits):
            fancy_types.append("BINARY_REPEATER")
        if builtin.get('doubles_ladder', True) and self.is_doubles_ladder(digits):
            fancy_types.append("DOUBLES_LADDER")
        if builtin.get('ladder_and_quad', True) and self.is_ladder_and_quad(digits):
            fancy_types.append("LADDER_AND_QUAD")

        # Custom patterns from config
        if self.cfg:
            for pattern in self.cfg.custom_patterns:
                if self.check_custom_pattern(pattern, digits, serial):
                    name = pattern.get('name', 'CUSTOM')
                    fancy_types.append(name)

        return fancy_types


# =============================================================================
# SCANNER FORMAT DETECTION
# =============================================================================

class ScannerFormatDetector:
    """Detects scanner naming conventions and pairs front/back images."""

    @staticmethod
    def detect_format(image_folder: Path) -> str:
        """
        Detect the scanner naming format.
        Returns: 'suffix' (has _b files) or 'sequential' (numbered pairs)
        """
        files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.jpeg"))

        # Check for _b suffix pattern
        for f in files:
            if '_b.' in f.name.lower() or '_b.' in f.stem.lower():
                return 'suffix'

        return 'sequential'

    @staticmethod
    def find_pairs_suffix(image_folder: Path) -> list[BillPair]:
        """
        Find front/back pairs using suffix naming (e.g., 0001.jpg + 0001_b.jpg)
        """
        files = sorted(image_folder.glob("*.jpg")) + sorted(image_folder.glob("*.jpeg"))
        pairs = []
        seen_bases = set()

        for f in files:
            # Skip back images in first pass
            if '_b.' in f.name.lower():
                continue

            base_name = f.stem
            back_patterns = [
                f.parent / f"{base_name}_b{f.suffix}",
                f.parent / f"{base_name}_B{f.suffix}",
            ]

            back_path = None
            for bp in back_patterns:
                if bp.exists():
                    back_path = bp
                    break

            if base_name not in seen_bases:
                seen_bases.add(base_name)
                pairs.append(BillPair(
                    front_path=f,
                    back_path=back_path,
                    stack_position=len(pairs) + 1
                ))

        return pairs

    @staticmethod
    def find_pairs_sequential(image_folder: Path) -> list[BillPair]:
        """
        Find front/back pairs using sequential numbering.
        Assumes files are sorted and alternating (front, back, front, back...).
        Will verify using YOLO detection later.
        """
        files = sorted(
            list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.jpeg")),
            key=lambda x: x.name
        )

        pairs = []
        i = 0
        while i < len(files):
            front = files[i]
            back = files[i + 1] if i + 1 < len(files) else None

            pairs.append(BillPair(
                front_path=front,
                back_path=back,
                stack_position=len(pairs) + 1
            ))
            i += 2

        return pairs

    @classmethod
    def find_pairs(cls, image_folder: Path) -> tuple[str, list[BillPair]]:
        """
        Detect format and find all front/back pairs.
        Returns: (format_name, list of BillPair)
        """
        fmt = cls.detect_format(image_folder)

        if fmt == 'suffix':
            pairs = cls.find_pairs_suffix(image_folder)
        else:
            pairs = cls.find_pairs_sequential(image_folder)

        return fmt, pairs


# =============================================================================
# PRODUCTION PROCESSOR
# =============================================================================

class ProductionProcessor:
    """Main processor for production pipeline."""

    def __init__(self, yolo_model_path: Path, reference_path: Path, use_gpu: bool = False, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()  # Use provided config or create default

        print(f"Loading YOLOv8 model: {yolo_model_path}")
        self.yolo_model = YOLO(str(yolo_model_path))

        print(f"Loading template aligner...")
        self.aligner = BillAligner(reference_path)

        print(f"Loading EasyOCR (GPU={use_gpu})...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)

        self.fancy_detector = FancyNumberDetector(cfg=self.cfg)
        print("Ready!\n")

    def count_serial_detections(self, image_path: Path) -> int:
        """Count how many serial numbers YOLO detects in an image."""
        img = cv2.imread(str(image_path))
        if img is None:
            return 0

        # Use lower confidence threshold to handle scans with colored backgrounds
        results = self.yolo_model(img, verbose=False, conf=0.1)
        count = 0
        for result in results:
            count += len(result.boxes)
        return count

    def is_front_image(self, image_path: Path) -> bool:
        """Determine if an image is a front (has serial numbers)."""
        return self.count_serial_detections(image_path) >= 1

    def verify_and_swap_pairs(self, pairs: list[BillPair]) -> list[BillPair]:
        """
        Verify front/back assignments and swap if needed.
        Front images have serial numbers, backs don't.
        """
        verified = []
        for pair in pairs:
            front_is_front = self.is_front_image(pair.front_path)
            back_is_front = pair.back_path and self.is_front_image(pair.back_path)

            # Swap if needed
            if not front_is_front and back_is_front:
                pair.front_path, pair.back_path = pair.back_path, pair.front_path
            elif not front_is_front and not back_is_front:
                # Neither has serials - might be two backs, mark as error
                pair.error = "No serial detected in either image"

            verified.append(pair)
        return verified

    def extract_serial_from_crop(self, crop_image) -> tuple[Optional[str], float]:
        """Extract serial number from a cropped region using OCR."""
        results = self.ocr_reader.readtext(
            crop_image,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*',
            detail=1
        )

        pattern = r'[A-Z]\d{8}[A-Z*]'

        # Common OCR letter/digit confusions
        corrections = {
            '0': ['O', 'Q', 'D'], '1': ['I', 'L'], '5': ['S'],
            '6': ['G'], '8': ['B'],
        }

        for (bbox, text, conf) in results:
            text_clean = re.sub(r'[^A-Z0-9*]', '', text.upper())

            match = re.search(pattern, text_clean)
            if match:
                return match.group(0), conf

            # Handle OCR errors - last digit misread as number (e.g., A→4, B→8)
            if re.match(r'^[A-Z]\d{9}$', text_clean):
                last_digit = text_clean[-1]
                if last_digit in corrections:
                    for letter in corrections[last_digit]:
                        corrected = text_clean[:-1] + letter
                        if re.match(pattern, corrected):
                            return corrected, conf

            # Handle OCR errors - first digit misread as number (e.g., G→6, B→8)
            if re.match(r'^\d{9}[A-Z*]$', text_clean):
                first_digit = text_clean[0]
                if first_digit in corrections:
                    for letter in corrections[first_digit]:
                        corrected = letter + text_clean[1:]
                        if re.match(pattern, corrected):
                            return corrected, conf

            # Handle OCR errors - BOTH first AND last misread as numbers
            if re.match(r'^\d{10}$', text_clean):
                first_digit = text_clean[0]
                last_digit = text_clean[-1]
                if first_digit in corrections and last_digit in corrections:
                    for first_letter in corrections[first_digit]:
                        for last_letter in corrections[last_digit]:
                            corrected = first_letter + text_clean[1:-1] + last_letter
                            if re.match(pattern, corrected):
                                return corrected, conf

        return None, 0

    def extract_serial(self, image_path: Path) -> tuple[Optional[str], float]:
        """Extract serial number from a bill image."""
        aligned_img = self.aligner.align_image(image_path)
        if aligned_img is None:
            return None, 0

        # Use lower confidence threshold to handle scans with colored backgrounds
        results = self.yolo_model(aligned_img, verbose=False, conf=0.1)
        serials_found = []
        h, w = aligned_img.shape[:2]

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Expand bounding box
                box_width = x2 - x1
                box_height = y2 - y1
                padding_x = int(box_width * 0.30)
                padding_y = int(box_height * 0.10)

                x1_exp = max(0, x1 - padding_x)
                y1_exp = max(0, y1 - padding_y)
                x2_exp = min(w, x2 + padding_x)
                y2_exp = min(h, y2 + padding_y)

                crop = aligned_img[y1_exp:y2_exp, x1_exp:x2_exp]
                serial, conf = self.extract_serial_from_crop(crop)

                if serial:
                    serials_found.append((serial, conf))

        if serials_found:
            return max(serials_found, key=lambda x: x[1])
        return None, 0

    def create_crop(self, image: np.ndarray, side: str, region: str) -> np.ndarray:
        """Create a percentage-based crop from an image."""
        h, w = image.shape[:2]
        crop_regions = self.cfg.crop_regions
        coords = crop_regions[side][region]

        x1 = int(coords['x'] * w)
        y1 = int(coords['y'] * h)
        x2 = int((coords['x'] + coords['w']) * w)
        y2 = int((coords['y'] + coords['h']) * h)

        return image[y1:y2, x1:x2]

    def generate_crops(self, pair: BillPair, output_dir: Path) -> list[Path]:
        """Generate crops for a fancy bill pair based on config."""
        crop_paths = []

        front_img = cv2.imread(str(pair.front_path))
        back_img = cv2.imread(str(pair.back_path)) if pair.back_path else None

        crop_order = self.cfg.crop_order
        jpeg_quality = self.cfg.jpeg_quality

        for i, (side, region) in enumerate(crop_order, 1):
            if side == 'front':
                img = front_img
            else:
                img = back_img
                if img is None:
                    continue

            crop = self.create_crop(img, side, region)
            filename = f"{pair.serial}_{i:02d}.jpg"
            crop_path = output_dir / filename

            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            crop_paths.append(crop_path)

        return crop_paths

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        verify_pairs: bool = True,
        crop_all: bool = False
    ) -> dict:
        """
        Process all bills in a directory.

        Args:
            input_dir: Directory containing scanned bills
            output_dir: Directory for fancy bill crops
            verify_pairs: Whether to verify front/back with YOLO (slower but more accurate)
            crop_all: If True, crop ALL bills regardless of fancy status

        Returns:
            Dictionary with processing results
        """
        start_time = time.time()

        # Detect format and find pairs
        print(f"Scanning directory: {input_dir}")
        scanner_format, pairs = ScannerFormatDetector.find_pairs(input_dir)
        print(f"Detected format: {scanner_format}")
        print(f"Found {len(pairs)} bill pairs\n")

        # Verify front/back assignments if requested
        if verify_pairs:
            print("Verifying front/back assignments...")
            pairs = self.verify_and_swap_pairs(pairs)
            print("Verification complete\n")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each pair
        print("Processing bills...")
        print("=" * 70)

        fancy_bills = []
        all_results = []
        non_fancy_files = []

        for pair in pairs:
            if pair.error:
                all_results.append({
                    'position': pair.stack_position,
                    'front_file': pair.front_path.name,
                    'back_file': pair.back_path.name if pair.back_path else '',
                    'serial': '',
                    'fancy_types': '',
                    'error': pair.error
                })
                continue

            # Extract serial
            serial, confidence = self.extract_serial(pair.front_path)
            pair.serial = serial
            pair.confidence = confidence

            if serial:
                # Check for fancy patterns (skip if --all flag)
                if crop_all:
                    pair.fancy_types = ["ALL"]
                    pair.is_fancy = True
                else:
                    fancy_types = self.fancy_detector.classify(serial)
                    pair.fancy_types = fancy_types
                    pair.is_fancy = len(fancy_types) > 0

                fancy_str = ", ".join(pair.fancy_types) if pair.fancy_types else ""
                status = f"[{fancy_str}]" if pair.fancy_types else ""
                print(f"#{pair.stack_position:3d}: {serial} {status}")

                if pair.is_fancy:
                    fancy_bills.append(pair)
                    # Generate crops
                    self.generate_crops(pair, output_dir)
                else:
                    # Track non-fancy files for potential cleanup
                    non_fancy_files.append(str(pair.front_path))
                    if pair.back_path:
                        non_fancy_files.append(str(pair.back_path))
            else:
                pair.error = "No serial detected"
                print(f"#{pair.stack_position:3d}: [ERROR] No serial detected")

            all_results.append({
                'position': pair.stack_position,
                'front_file': pair.front_path.name,
                'back_file': pair.back_path.name if pair.back_path else '',
                'serial': serial or '',
                'fancy_types': ", ".join(pair.fancy_types),
                'confidence': f"{confidence:.2f}",
                'is_fancy': pair.is_fancy,
                'error': pair.error or ''
            })

        # Calculate stats
        total_time = time.time() - start_time
        total_bills = len(pairs)
        successful = sum(1 for p in pairs if p.serial)
        fancy_count = len(fancy_bills)

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total bills processed: {total_bills}")
        print(f"Successfully extracted: {successful} ({successful/total_bills*100:.1f}%)")
        print(f"Fancy numbers found: {fancy_count}")
        print(f"Processing time: {total_time:.1f} seconds")
        print(f"Rate: {total_bills/total_time*60:.1f} bills/minute")

        # Show fancy bill positions
        if fancy_bills:
            positions = [str(p.stack_position) for p in fancy_bills]
            print(f"\nFancy bills at stack positions: {', '.join(positions)}")
            print("\nFancy bills found:")
            for p in fancy_bills:
                print(f"  #{p.stack_position}: {p.serial} [{', '.join(p.fancy_types)}]")

        # Save results
        self._save_results(input_dir, output_dir, all_results, fancy_bills, non_fancy_files)

        return {
            'total': total_bills,
            'successful': successful,
            'fancy_count': fancy_count,
            'fancy_positions': [p.stack_position for p in fancy_bills],
            'time': total_time
        }

    def _save_results(
        self,
        input_dir: Path,
        output_dir: Path,
        all_results: list,
        fancy_bills: list,
        non_fancy_files: list
    ):
        """Save all output files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Full results CSV
        csv_path = input_dir / f"results_{timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'position', 'front_file', 'back_file', 'serial',
                'fancy_types', 'confidence', 'is_fancy', 'error'
            ])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved: {csv_path}")

        # 2. Fancy bill summary
        summary_path = input_dir / f"summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("FANCY BILL SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            if fancy_bills:
                f.write("Stack positions with fancy bills:\n")
                positions = [str(p.stack_position) for p in fancy_bills]
                f.write(f"  {', '.join(positions)}\n\n")

                f.write("Details:\n")
                for p in fancy_bills:
                    f.write(f"  Position #{p.stack_position}: {p.serial}\n")
                    f.write(f"    Types: {', '.join(p.fancy_types)}\n")
                    f.write(f"    Crops: {output_dir / p.serial}_01.jpg ... _10.jpg\n\n")
            else:
                f.write("No fancy bills found.\n")
        print(f"Summary saved: {summary_path}")

        # 3. Non-fancy files list (for optional cleanup)
        cleanup_path = input_dir / f"non_fancy_files_{timestamp}.txt"
        with open(cleanup_path, 'w') as f:
            f.write("# Non-fancy bill scan files\n")
            f.write("# These can be deleted to save space\n")
            f.write("# Review before deleting!\n\n")
            for filepath in non_fancy_files:
                f.write(f"{filepath}\n")
        print(f"Cleanup list saved: {cleanup_path}")
        print(f"  ({len(non_fancy_files)} files can be deleted to save space)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Production pipeline for dollar bill processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_production.py ./scans --output ./fancy_bills
  python process_production.py ./scans --output ./fancy_bills --config ./config.yaml
  python process_production.py ./scans --output ./fancy_bills --reference ./ref.jpg
  python process_production.py ./scans --output ./fancy_bills --gpu
        """
    )

    parser.add_argument('input_dir', type=Path,
                        help='Directory containing scanned bill images')
    parser.add_argument('--output', '-o', type=Path, default=Path('./fancy_bills'),
                        help='Output directory for fancy bill crops (default: ./fancy_bills)')
    parser.add_argument('--config', '-c', type=Path, default=None,
                        help='Path to config.yaml (default: config.yaml in script directory)')
    parser.add_argument('--model', '-m', type=Path, default=None,
                        help='Path to YOLO model (default: best.pt in script directory)')
    parser.add_argument('--reference', '-r', type=Path, default=None,
                        help='Reference image for alignment (default: first front image)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for processing')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip front/back verification (faster but less accurate)')
    parser.add_argument('--all', action='store_true',
                        help='Crop ALL bills (skip fancy detection, treat everything as fancy)')

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1

    # Find and load config
    script_dir = Path(__file__).parent
    config_path = args.config or script_dir / "config.yaml"
    patterns_path = script_dir / "patterns.txt"

    cfg = Config(
        config_path if config_path.exists() else None,
        patterns_path if patterns_path.exists() else None
    )

    # Show custom patterns if any
    if cfg.custom_patterns:
        print(f"Custom patterns: {[p.get('name', 'unnamed') for p in cfg.custom_patterns]}")

    # Find model
    model_path = args.model or script_dir / "best.pt"
    if not model_path.exists():
        print(f"Error: YOLO model not found: {model_path}")
        return 1

    # Find reference image (priority: --reference flag > first image in input dir)
    # Note: Using first image from input ensures reference matches scanner/resolution
    if args.reference:
        reference_path = args.reference
        print(f"Using reference: {reference_path}")
    else:
        # Use first image in input directory (best for matching scanner characteristics)
        images = sorted(list(args.input_dir.glob("*.jpg")) + list(args.input_dir.glob("*.jpeg")))
        if not images:
            print(f"Error: No images found in {args.input_dir}")
            return 1
        reference_path = images[0]
        print(f"Using first image as reference: {reference_path.name}")

    # Initialize and run
    processor = ProductionProcessor(model_path, reference_path, use_gpu=args.gpu, cfg=cfg)

    if args.all:
        print("Mode: Processing ALL bills (--all flag)")

    results = processor.process_directory(
        args.input_dir,
        args.output,
        verify_pairs=not args.no_verify,
        crop_all=args.all
    )

    print(f"\nCrops saved to: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
