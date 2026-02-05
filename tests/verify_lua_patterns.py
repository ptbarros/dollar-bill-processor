#!/usr/bin/env python3
"""
Verification script for Lua patterns.

Tests that:
1. Pattern engine loads all patterns successfully
2. Each Lua pattern matches its expected examples
3. Pattern metadata (odds, price) is available
4. Visualization data is valid
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pattern_engine_v3 import PatternEngineV3


def test_pattern_loading():
    """Test that all patterns load successfully."""
    print("=" * 70)
    print("Testing Pattern Loading")
    print("=" * 70)

    engine = PatternEngineV3()

    lua_count = len(engine.lua_patterns)
    total = len(engine.get_all_patterns())

    print(f"Loaded {lua_count} Lua patterns")
    print(f"Total unique patterns: {total}")

    # Show library breakdown
    libs = {}
    for name, info in engine.lua_patterns.items():
        lib = info.library
        if lib not in libs:
            libs[lib] = 0
        libs[lib] += 1

    print("\nPatterns by library:")
    for lib, count in sorted(libs.items()):
        print(f"  {lib}: {count}")

    return lua_count > 0


def test_example_matching():
    """Test that Lua patterns match their header examples."""
    print("\n" + "=" * 70)
    print("Testing Example Matching")
    print("=" * 70)

    engine = PatternEngineV3()

    passed = 0
    failed = 0
    skipped = 0

    for pattern_name, info in sorted(engine.lua_patterns.items()):
        # Only test enabled patterns
        if not info.enabled:
            continue

        examples = info.examples

        if not examples:
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

    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped (no examples)")
    return failed == 0


def test_metadata():
    """Test that patterns have metadata (odds, price)."""
    print("\n" + "=" * 70)
    print("Testing Metadata")
    print("=" * 70)

    engine = PatternEngineV3()

    with_odds = 0
    with_price = 0
    total = len(engine.lua_patterns)

    for name, info in engine.lua_patterns.items():
        if info.odds:
            with_odds += 1
        if info.price:
            with_price += 1

    print(f"Patterns with odds: {with_odds}/{total}")
    print(f"Patterns with price: {with_price}/{total}")

    # Check specific patterns have expected data
    radar_info = engine.get_pattern_info("RADAR")
    if radar_info:
        print(f"\nRADAR pattern info:")
        print(f"  Tier: {radar_info.get('tier')}")
        print(f"  Odds: {radar_info.get('odds')}")
        print(f"  Price: {radar_info.get('price')}")

        if radar_info.get('odds') and radar_info.get('price'):
            print("  Status: OK")
            return True
        else:
            print("  Status: MISSING DATA")
            return False
    else:
        print("RADAR pattern not found!")
        return False


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
    results.append(("Metadata", test_metadata()))
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
