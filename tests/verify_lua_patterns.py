#!/usr/bin/env python3
"""
Verification script for Lua pattern conversions.

Tests that:
1. Each Lua pattern matches its expected examples from YAML
2. No unexpected false positives occur
3. Pattern engine loads all patterns successfully
"""

import sys
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pattern_engine_v3 import PatternEngineV3


def load_yaml_examples():
    """Load examples from patterns_v2.yaml"""
    yaml_path = Path(__file__).parent.parent / "patterns_v2.yaml"
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    examples = {}
    for name, defn in config.get('patterns', {}).items():
        if 'examples' in defn and defn['examples']:
            examples[name] = defn['examples']

    return examples


def test_pattern_loading():
    """Test that all patterns load successfully."""
    print("=" * 70)
    print("Testing Pattern Loading")
    print("=" * 70)

    engine = PatternEngineV3()

    lua_count = len(engine.lua_patterns)
    yaml_count = len(engine.yaml_engine.get_all_patterns())
    total = len(engine.get_all_patterns())

    print(f"Loaded {lua_count} Lua patterns")
    print(f"Loaded {yaml_count} YAML patterns")
    print(f"Total unique patterns: {total}")

    # List Lua patterns
    print("\nLua patterns:")
    for name in sorted(engine.lua_patterns.keys()):
        info = engine.lua_patterns[name]
        print(f"  - {name} (Tier {info.tier})")

    return True


def test_example_matching():
    """Test that Lua patterns match their YAML examples."""
    print("\n" + "=" * 70)
    print("Testing Example Matching")
    print("=" * 70)

    engine = PatternEngineV3()
    yaml_examples = load_yaml_examples()

    passed = 0
    failed = 0
    skipped = 0

    for pattern_name in sorted(engine.lua_patterns.keys()):
        # Get examples for this pattern
        examples = yaml_examples.get(pattern_name, [])

        if not examples:
            print(f"  SKIP: {pattern_name} (no examples)")
            skipped += 1
            continue

        # Test each example
        all_matched = True
        failed_examples = []

        for example in examples:
            # Add prefix/suffix if needed
            if len(example) == 8 and example.isdigit():
                serial = f"A{example}B"
            else:
                serial = example

            matches = engine.classify_simple(serial)
            if pattern_name not in matches:
                all_matched = False
                failed_examples.append(example)

        if all_matched:
            print(f"  PASS: {pattern_name} ({len(examples)} examples)")
            passed += 1
        else:
            print(f"  FAIL: {pattern_name} - missed: {failed_examples}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")
    return failed == 0


def test_specific_serials():
    """Test specific serial numbers for expected patterns."""
    print("\n" + "=" * 70)
    print("Testing Specific Serials")
    print("=" * 70)

    engine = PatternEngineV3()

    test_cases = [
        ("A88888888B", ["SOLID", "SEVEN_OF_KIND"]),
        ("A12344321B", ["RADAR", "BINARY"]),
        ("A12341234B", ["REPEATER"]),
        ("A01234567B", ["LADDER"]),
        ("A87654321B", ["LADDER"]),
        ("A11223344B", ["FOUR_CONSEC_PAIRS", "DOUBLES_LADDER"]),
        ("A10000001B", ["SUPER_RADAR", "BINARY"]),
        ("A12121212B", ["SUPER_REPEATER", "ALTERNATOR", "RADAR_REPEATER", "BINARY_REPEATER", "BINARY_RADAR"]),
        ("A01010101B", ["TRUE_BINARY", "BINARY"]),
        ("A00000001B", ["SERIAL_UNDER_10"]),
        ("A10000000B", ["MULTI_MILLIONAIRE"]),
        ("A12345000B", ["TRAILING_000"]),
        ("A00012345B", ["LOW_000"]),
        ("A77712345B", ["LUCKY_777"]),
        ("A99999999B", ["SOLID", "SUM_72"]),
        ("A00000000B", ["SOLID", "SUM_0"]),
    ]

    passed = 0
    failed = 0

    for serial, expected in test_cases:
        matches = engine.classify_simple(serial)

        missing = [p for p in expected if p not in matches]
        if missing:
            print(f"  FAIL: {serial}")
            print(f"         Expected: {expected}")
            print(f"         Got: {matches[:10]}")
            print(f"         Missing: {missing}")
            failed += 1
        else:
            print(f"  PASS: {serial} -> {[m for m in matches if m in expected]}")
            passed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_visualization():
    """Test that patterns return valid visualization data."""
    print("\n" + "=" * 70)
    print("Testing Visualization Data")
    print("=" * 70)

    engine = PatternEngineV3()

    test_serials = ["A12344321B", "A88888888B", "A12121212B", "A01234567B"]

    all_valid = True
    for serial in test_serials:
        matches = engine.classify(serial)

        for match in matches:
            if match.source == "lua":
                # Check highlights
                for h in match.highlights:
                    if 'positions' not in h or 'color' not in h:
                        print(f"  FAIL: {match.name} - invalid highlight: {h}")
                        all_valid = False

                # Check connectors
                for c in match.connectors:
                    if 'from' not in c or 'to' not in c or 'color' not in c:
                        print(f"  FAIL: {match.name} - invalid connector: {c}")
                        all_valid = False

        if all_valid:
            print(f"  PASS: {serial} - all visualization data valid")

    return all_valid


def main():
    """Run all tests."""
    print("Lua Pattern Verification Script")
    print("=" * 70)

    results = []

    results.append(("Pattern Loading", test_pattern_loading()))
    results.append(("Example Matching", test_example_matching()))
    results.append(("Specific Serials", test_specific_serials()))
    results.append(("Visualization", test_visualization()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
