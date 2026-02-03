"""
Test script for the DataFile feature in Lua patterns.

Run with: python tests/test_datafile_feature.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pattern_engine_v3 import PatternEngineV3


def test_datafile_loading():
    """Test that data files are loaded correctly."""
    print("Testing DataFile Loading")
    print("=" * 60)

    engine = PatternEngineV3()

    # Check if KNOWN_SERIALS pattern exists and has data loaded
    if 'KNOWN_SERIALS' not in engine.lua_patterns:
        print("KNOWN_SERIALS pattern not found - creating test files...")
        print("Please ensure patterns/user/known_serials.lua and .csv exist")
        return False

    info = engine.lua_patterns['KNOWN_SERIALS']
    print(f"Pattern: {info.name}")
    print(f"  data_file: {info.data_file}")
    print(f"  data loaded: {info.data is not None}")
    print(f"  data_by_key loaded: {info.data_by_key is not None}")

    if info.data:
        print(f"  rows in data: {len(info.data)}")
        print(f"  first row: {info.data[0]}")

    if info.data_by_key:
        print(f"  keys in data_by_key: {list(info.data_by_key.keys())}")

    return info.data is not None and info.data_by_key is not None


def test_pattern_matching():
    """Test that patterns with data files match correctly."""
    print("\nTesting Pattern Matching with Data")
    print("=" * 60)

    engine = PatternEngineV3()

    # Test serials that should match
    test_cases = [
        ("A12345678B", True, "Perfect ladder"),
        ("A88888888B", True, "Solid 8s"),
        ("A12344321B", True, "Radar example"),
        ("A99999999B", False, "Not in known list"),
        ("A11223344B", False, "Not in known list"),
    ]

    all_passed = True
    for serial, should_match, expected_desc in test_cases:
        matches = engine.classify(serial)
        match_names = [m.name for m in matches]
        matched = 'KNOWN_SERIALS' in match_names

        status = "PASS" if matched == should_match else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(f"{status}: {serial} - {expected_desc}")
        print(f"       Expected match: {should_match}, Got: {matched}")

        # Show message if matched
        if matched:
            for m in matches:
                if m.name == 'KNOWN_SERIALS':
                    print(f"       Message: {m.message}")

    return all_passed


def test_missing_datafile():
    """Test graceful handling of missing data files."""
    print("\nTesting Missing DataFile Handling")
    print("=" * 60)

    # Create a temporary pattern with a missing data file
    from pattern_sandbox import PatternSandbox, create_context

    script = '''--[[
Pattern: TEST_MISSING
Description: Test missing data file
Tier: 10
DataFile: nonexistent.csv
--]]

function match(ctx)
    if not ctx.data_by_key then
        return {matched = false}
    end
    return {matched = true}
end
'''

    sandbox = PatternSandbox()
    ctx = create_context("A12345678B")
    # Data won't be injected since file doesn't exist
    result = sandbox.execute(script, ctx)

    print(f"Pattern with missing data file:")
    print(f"  executed successfully: {result.success}")
    print(f"  matched: {result.matched}")
    print(f"  (should be False because ctx.data_by_key is nil)")

    return result.success and not result.matched


def test_data_injection():
    """Test that data is properly injected into Lua context."""
    print("\nTesting Data Injection")
    print("=" * 60)

    from pattern_sandbox import PatternSandbox, create_context

    # Load helpers
    sandbox = PatternSandbox()
    helpers_path = Path(__file__).parent.parent / "patterns" / "lib" / "helpers.lua"
    if helpers_path.exists():
        with open(helpers_path) as f:
            sandbox.load_helpers(f.read())

    script = '''
function match(ctx)
    -- Test that data is accessible
    if not ctx.data then
        return {matched = false, message = "No data"}
    end

    -- Test iterating over data
    local count = 0
    for _, row in ipairs(ctx.data) do
        count = count + 1
    end

    -- Test data_by_key lookup
    local found = ctx.data_by_key and ctx.data_by_key["test_key"]

    return {
        matched = true,
        message = "Rows: " .. count .. ", Key lookup: " .. tostring(found ~= nil)
    }
end
'''

    ctx = create_context("A12345678B")
    # Manually inject test data
    ctx['data'] = [
        {'key': 'test_key', 'value': 'one'},
        {'key': 'other', 'value': 'two'},
    ]
    ctx['data_by_key'] = {'test_key': {'key': 'test_key', 'value': 'one'}}

    result = sandbox.execute(script, ctx)

    print(f"Data injection test:")
    print(f"  success: {result.success}")
    print(f"  matched: {result.matched}")
    print(f"  message: {result.message}")

    return result.success and result.matched and "Rows: 2" in result.message


def main():
    print("DataFile Feature Tests")
    print("=" * 60)
    print()

    tests = [
        ("DataFile Loading", test_datafile_loading),
        ("Pattern Matching", test_pattern_matching),
        ("Missing DataFile", test_missing_datafile),
        ("Data Injection", test_data_injection),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
        print()

    print("=" * 60)
    print("Summary:")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")

    all_passed = all(p for _, p in results)
    print()
    print("Overall:", "PASS" if all_passed else "FAIL")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
