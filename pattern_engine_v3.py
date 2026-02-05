"""
Pattern Engine v3 - Hybrid Lua/YAML Pattern System

Supports both:
- Legacy YAML patterns (via pattern_engine_v2)
- Lua script patterns (via pattern_sandbox)

Script patterns can provide rich visualization with custom highlights and connectors.
"""

import re
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from pattern_engine_v2 import PatternEngine as PatternEngineV2, PatternMatch
from pattern_sandbox import PatternSandbox, create_context, LuaExecutionResult


@dataclass
class PatternMatchV3(PatternMatch):
    """Extended pattern match with visualization data."""
    highlights: list = field(default_factory=list)
    connectors: list = field(default_factory=list)
    group_boxes: list = field(default_factory=list)  # Boxes spanning multiple digits
    message: str = ""
    source: str = "yaml"  # "yaml" or "lua"


@dataclass
class LuaPatternInfo:
    """Information about a loaded Lua pattern."""
    name: str
    description: str
    tier: int
    script: str
    file_path: Path
    enabled: bool = True
    display_name: str = ""        # User-friendly name for GUI (falls back to name if empty)
    examples: list = field(default_factory=list)
    odds: str = ""
    price: str = ""
    data_file: str = ""           # Relative path to external data file (CSV/JSON)
    data: Any = None              # Loaded data (list of dicts for CSV, any for JSON)
    data_by_key: dict = None      # Dict keyed by first column (CSV only)


class PatternEngineV3:
    """
    Hybrid pattern engine supporting YAML and Lua patterns.

    Usage:
        engine = PatternEngineV3()
        matches = engine.classify("A12344321B")
        highlights = engine.get_digit_highlights("A12344321B", ["RADAR"])
    """

    def __init__(self, config_path: Path = None, patterns_dir: Path = None):
        """
        Initialize the hybrid engine.

        Args:
            config_path: Path to patterns_v2.yaml (default: auto-detect)
            patterns_dir: Path to patterns/ directory (default: auto-detect)
        """
        # Base directory
        base_dir = Path(__file__).parent

        # Initialize YAML engine (v2)
        self.yaml_engine = PatternEngineV2(config_path)

        # Set up patterns directory
        if patterns_dir is None:
            self.patterns_dir = base_dir / "patterns"
        else:
            self.patterns_dir = Path(patterns_dir)

        # Initialize Lua sandbox
        self.sandbox = PatternSandbox()
        self._load_helpers()

        # Load Lua patterns
        self.lua_patterns: Dict[str, LuaPatternInfo] = {}
        self._load_lua_patterns()

        # Cache for user config path
        self.user_patterns_dir = self.patterns_dir / "user"

    def _load_helpers(self):
        """Load helper functions into sandbox."""
        helpers_path = self.patterns_dir / "lib" / "helpers.lua"
        if helpers_path.exists():
            with open(helpers_path, 'r') as f:
                helpers_code = f.read()
            self.sandbox.load_helpers(helpers_code)

    def _load_lua_patterns(self):
        """Load all Lua pattern scripts from all library directories."""
        self.lua_patterns.clear()

        # Skip these directories (not pattern libraries)
        skip_dirs = {'lib', 'data', '__pycache__'}

        # Scan all subdirectories under patterns/
        if self.patterns_dir.exists():
            for subdir in sorted(self.patterns_dir.iterdir()):
                if not subdir.is_dir() or subdir.name in skip_dirs:
                    continue

                # core is loaded first, everything else is considered "user" (can override)
                is_user = subdir.name != 'core'

                for lua_file in subdir.glob("*.lua"):
                    self._load_lua_pattern(lua_file, is_user=is_user)

    def _load_lua_pattern(self, file_path: Path, is_user: bool = False):
        """Load a single Lua pattern from file."""
        try:
            with open(file_path, 'r') as f:
                script = f.read()

            # Parse metadata from header comment
            metadata = self._parse_lua_metadata(script)

            # Use filename as pattern name if not in metadata
            name = metadata.get('name', file_path.stem.upper())

            # Validate syntax
            valid, error = self.sandbox.validate_syntax(script)
            if not valid:
                print(f"Warning: Syntax error in {file_path}: {error}")
                return

            # Create pattern info
            info = LuaPatternInfo(
                name=name,
                description=metadata.get('description', ''),
                tier=int(metadata.get('tier', 10)),
                script=script,
                file_path=file_path,
                enabled=metadata.get('enabled', True),
                display_name=metadata.get('displayname', ''),
                examples=metadata.get('examples', []),
                odds=metadata.get('odds', ''),
                price=metadata.get('price', '')
            )

            # Load external data file if specified
            data_file = metadata.get('datafile', '')
            if data_file:
                data, data_by_key = self._load_data_file(data_file, file_path)
                info.data_file = data_file
                info.data = data
                info.data_by_key = data_by_key

            self.lua_patterns[name] = info

        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")

    def _parse_lua_metadata(self, script: str) -> dict:
        """Parse metadata from Lua script header comment."""
        metadata = {}

        # Look for header comment block --[[ ... --]]
        match = re.search(r'--\[\[(.*?)--\]\]', script, re.DOTALL)
        if not match:
            # Try line comments at the start
            lines = script.strip().split('\n')
            for line in lines:
                if not line.strip().startswith('--'):
                    break
                # Parse "-- Key: Value" format
                m = re.match(r'--\s*(\w+):\s*(.+)', line)
                if m:
                    key = m.group(1).lower()
                    value = m.group(2).strip()
                    metadata[key] = self._parse_metadata_value(key, value)
            return metadata

        header = match.group(1)

        # Parse key: value pairs
        for line in header.split('\n'):
            m = re.match(r'\s*(\w+):\s*(.+)', line)
            if m:
                key = m.group(1).lower()
                value = m.group(2).strip()
                metadata[key] = self._parse_metadata_value(key, value)

        return metadata

    def _parse_metadata_value(self, key: str, value: str) -> Any:
        """Parse a metadata value based on its key."""
        if key == 'tier':
            try:
                return int(value)
            except ValueError:
                return 10
        elif key == 'examples':
            # Parse JSON-like list ["a", "b"]
            try:
                import json
                return json.loads(value)
            except:
                return [value]
        elif key == 'enabled':
            return value.lower() in ('true', 'yes', '1')
        elif key == 'datafile':
            # Keep as string path
            return value
        else:
            return value

    def _load_data_file(self, data_file: str, lua_file_path: Path) -> tuple:
        """
        Load external data file for a pattern.

        Path resolution:
        1. If starts with 'data/': look in patterns/data/
        2. Otherwise: resolve relative to the .lua file's directory

        Args:
            data_file: Relative path to data file from pattern header
            lua_file_path: Path to the Lua pattern file

        Returns:
            (data, data_by_key) tuple, or (None, None) on error
        """
        # Resolve path
        if data_file.startswith('data/'):
            file_path = self.patterns_dir / data_file
        else:
            file_path = lua_file_path.parent / data_file

        if not file_path.exists():
            print(f"Warning: Data file not found: {file_path}")
            return None, None

        try:
            suffix = file_path.suffix.lower()
            if suffix == '.csv':
                return self._load_csv(file_path)
            elif suffix == '.json':
                return self._load_json(file_path)
            else:
                print(f"Warning: Unsupported data file format: {suffix}")
                return None, None
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            return None, None

    def _load_csv(self, file_path: Path) -> tuple:
        """
        Load CSV file as list of dicts with key lookup.

        Returns:
            (rows, data_by_key) where:
            - rows: list of dicts with column names as keys
            - data_by_key: dict keyed by first column value
        """
        import csv
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        # Build key lookup from first column
        data_by_key = {}
        if rows and fieldnames:
            first_col = fieldnames[0]
            for row in rows:
                key = row.get(first_col, '')
                if key:
                    data_by_key[key] = row

        return rows, data_by_key

    def _load_json(self, file_path: Path) -> tuple:
        """
        Load JSON file.

        Returns:
            (data, None) - JSON data can be any structure, no automatic key lookup
        """
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data, None

    def reload(self):
        """Reload all patterns (both YAML and Lua)."""
        self.yaml_engine.reload()
        self._load_lua_patterns()

    def classify(self, serial: str, metadata: dict = None) -> List[PatternMatchV3]:
        """
        Classify a serial number against all patterns.

        Args:
            serial: Serial number to classify
            metadata: Optional metadata dict for pattern matching

        Returns:
            List of PatternMatchV3 sorted by tier
        """
        matches = []

        # Get digits for validation
        digits = ''.join(c for c in serial if c.isdigit())
        if len(digits) != 8:
            return []

        # Run Lua patterns first (they take precedence)
        base_ctx = create_context(serial, metadata)
        for name, info in self.lua_patterns.items():
            if not info.enabled:
                continue

            try:
                # Create context with pattern's data injected
                ctx = base_ctx.copy()
                if info.data is not None:
                    ctx['data'] = info.data
                if info.data_by_key is not None:
                    ctx['data_by_key'] = info.data_by_key

                result = self.sandbox.execute(info.script, ctx)
                if result.success and result.matched:
                    matches.append(PatternMatchV3(
                        name=name,
                        description=info.description,
                        tier=info.tier,
                        highlights=result.highlights,
                        connectors=result.connectors,
                        group_boxes=result.group_boxes,
                        message=result.message,
                        source="lua"
                    ))
            except Exception as e:
                # Log but don't fail on individual pattern errors
                pass

        # Run YAML patterns (skip if Lua file exists for that pattern name)
        yaml_matches = self.yaml_engine.classify(serial, metadata)
        for m in yaml_matches:
            # Skip YAML pattern if a Lua file exists for this pattern name
            # This prevents loose YAML patterns from overriding stricter Lua logic
            if m.name not in self.lua_patterns:
                # Get visualization from YAML engine
                highlights_data = self.yaml_engine.get_digit_highlights(serial, [m.name])

                matches.append(PatternMatchV3(
                    name=m.name,
                    description=m.description,
                    tier=m.tier,
                    highlights=highlights_data.get('highlights', []),
                    connectors=highlights_data.get('connectors', []),
                    source="yaml"
                ))

        # Sort by tier
        matches.sort(key=lambda m: (m.tier, m.name))
        return matches

    def classify_simple(self, serial: str, metadata: dict = None) -> List[str]:
        """Return just pattern names."""
        return [m.name for m in self.classify(serial, metadata)]

    def get_digit_highlights(self, serial: str, pattern_names: List[str]) -> dict:
        """
        Get combined highlight and connector data for specified patterns.

        Args:
            serial: Serial number
            pattern_names: List of pattern names to get highlights for

        Returns:
            Dict with 'highlights', 'connectors', and 'group_boxes' lists
        """
        digits = ''.join(c for c in serial if c.isdigit())
        if len(digits) != 8:
            return {'highlights': [], 'connectors': [], 'group_boxes': []}

        # Initialize per-position highlights
        all_highlights = []
        for i, d in enumerate(digits):
            all_highlights.append({
                'position': i,
                'digit': d,
                'highlights': []
            })

        all_connectors = []
        all_group_boxes = []

        for pattern_name in pattern_names:
            # Check if it's a Lua pattern
            if pattern_name in self.lua_patterns:
                info = self.lua_patterns[pattern_name]
                if info.enabled:
                    ctx = create_context(serial)
                    # Inject pattern's data if available
                    if info.data is not None:
                        ctx['data'] = info.data
                    if info.data_by_key is not None:
                        ctx['data_by_key'] = info.data_by_key
                    result = self.sandbox.execute(info.script, ctx)
                    if result.success and result.matched:
                        # Merge Lua highlights
                        self._merge_lua_highlights(all_highlights, result.highlights, pattern_name)
                        # Add connectors
                        for conn in result.connectors:
                            conn['pattern'] = pattern_name
                            all_connectors.append(conn)
                        # Add group boxes
                        for gb in result.group_boxes:
                            gb['pattern'] = pattern_name
                            all_group_boxes.append(gb)
            else:
                # Fall back to YAML engine
                yaml_data = self.yaml_engine.get_digit_highlights(serial, [pattern_name])
                # Merge YAML highlights
                for h in yaml_data.get('highlights', []):
                    pos = h['position']
                    for hl in h.get('highlights', []):
                        all_highlights[pos]['highlights'].append(hl)
                # Add connectors
                all_connectors.extend(yaml_data.get('connectors', []))

        return {'highlights': all_highlights, 'connectors': all_connectors, 'group_boxes': all_group_boxes}

    def _merge_lua_highlights(self, all_highlights: list, lua_highlights: list, pattern_name: str):
        """Merge Lua script highlights into the per-position format."""
        for h in lua_highlights:
            positions = h.get('positions', [])
            color = h.get('color', 'gray')
            label = h.get('label', '')

            for pos in positions:
                if 0 <= pos < 8:
                    all_highlights[pos]['highlights'].append({
                        'pattern': pattern_name,
                        'color': color,
                        'reason': label or f'{pattern_name} match'
                    })

    def execute_pattern(self, pattern_name: str, serial: str, metadata: dict = None) -> LuaExecutionResult:
        """
        Execute a specific pattern and get detailed result.

        Args:
            pattern_name: Name of the pattern to execute
            serial: Serial number to test
            metadata: Optional metadata

        Returns:
            LuaExecutionResult with full details
        """
        if pattern_name in self.lua_patterns:
            info = self.lua_patterns[pattern_name]
            ctx = create_context(serial, metadata)
            # Inject pattern's data if available
            if info.data is not None:
                ctx['data'] = info.data
            if info.data_by_key is not None:
                ctx['data_by_key'] = info.data_by_key
            return self.sandbox.execute(info.script, ctx)
        else:
            # For YAML patterns, create a pseudo-result
            matches = self.yaml_engine.classify(serial, metadata)
            matched = any(m.name == pattern_name for m in matches)
            return LuaExecutionResult(
                success=True,
                matched=matched,
                message="YAML pattern" if matched else ""
            )

    def test_script(self, script: str, serial: str, metadata: dict = None) -> LuaExecutionResult:
        """
        Test a pattern script without saving it.

        Args:
            script: Lua script code
            serial: Serial number to test
            metadata: Optional metadata

        Returns:
            LuaExecutionResult with execution details
        """
        ctx = create_context(serial, metadata)
        return self.sandbox.execute(script, ctx)

    def validate_script(self, script: str) -> tuple:
        """
        Validate Lua script syntax.

        Returns:
            (is_valid, error_message)
        """
        return self.sandbox.validate_syntax(script)

    def save_user_pattern(self, name: str, script: str, description: str = "",
                          tier: int = 5, examples: list = None,
                          library: str = "user", display_name: str = "") -> bool:
        """
        Save a user pattern to a library directory.

        Args:
            name: Pattern name (will be converted to filename)
            script: Lua script code
            description: Pattern description
            tier: Pattern tier (1-10)
            examples: Example serial numbers
            library: Library directory name (default: "user")
            display_name: User-friendly name for GUI display

        Returns:
            True if saved successfully
        """
        # Determine the target directory based on library
        if library == "user":
            target_dir = self.user_patterns_dir
        else:
            target_dir = self.patterns_dir / library

        # Ensure directory exists
        target_dir.mkdir(parents=True, exist_ok=True)

        # Create filename from pattern name
        filename = name.lower().replace(' ', '_') + '.lua'
        file_path = target_dir / filename

        # Build header
        header_lines = [
            '--[[',
            f'Pattern: {name}',
        ]
        if display_name:
            header_lines.append(f'DisplayName: {display_name}')
        header_lines.extend([
            f'Description: {description}',
            f'Tier: {tier}',
        ])
        if examples:
            import json
            header_lines.append(f'Examples: {json.dumps(examples)}')
        header_lines.append('--]]')
        header = '\n'.join(header_lines)

        # Combine header and script
        full_script = f"{header}\n\n{script}"

        try:
            with open(file_path, 'w') as f:
                f.write(full_script)

            # Reload to pick up the new pattern
            # Consider non-core libraries as "user" patterns for enable/disable purposes
            is_user = library != 'core'
            self._load_lua_pattern(file_path, is_user=is_user)
            return True

        except Exception as e:
            print(f"Error saving pattern: {e}")
            return False

    def delete_user_pattern(self, name: str) -> bool:
        """
        Delete a user pattern.

        Args:
            name: Pattern name

        Returns:
            True if deleted successfully
        """
        if name not in self.lua_patterns:
            return False

        info = self.lua_patterns[name]

        # Only allow deleting user patterns
        if not str(info.file_path).startswith(str(self.user_patterns_dir)):
            return False

        try:
            info.file_path.unlink()
            del self.lua_patterns[name]
            return True
        except Exception as e:
            print(f"Error deleting pattern: {e}")
            return False

    def get_pattern_info(self, name: str) -> Optional[dict]:
        """Get info about a pattern (Lua or YAML)."""
        if name in self.lua_patterns:
            info = self.lua_patterns[name]
            result = {
                'name': info.name,
                'description': info.description,
                'tier': info.tier,
                'enabled': info.enabled,
                'source': 'lua',
                'examples': info.examples,
                'odds': info.odds,
                'price': info.price,
                'price_range': info.price,
                'script': info.script
            }
            # Include data file info if present
            if info.data_file:
                result['data_file'] = info.data_file
                result['data_loaded'] = info.data is not None
            return result
        else:
            yaml_info = self.yaml_engine.get_pattern_info(name)
            if yaml_info:
                yaml_info['source'] = 'yaml'
            return yaml_info

    def get_all_patterns(self) -> dict:
        """Get all patterns (combined Lua and YAML)."""
        patterns = {}

        # YAML patterns first
        for name, defn in self.yaml_engine.get_all_patterns().items():
            patterns[name] = defn
            patterns[name]['source'] = 'yaml'

        # Lua patterns override YAML
        for name, info in self.lua_patterns.items():
            patterns[name] = {
                'name': info.name,
                'description': info.description,
                'tier': info.tier,
                'enabled': info.enabled,
                'source': 'lua',
                'examples': info.examples,
            }

        return patterns

    def get_lua_patterns(self) -> Dict[str, LuaPatternInfo]:
        """Get all Lua patterns."""
        return self.lua_patterns.copy()

    def get_user_patterns(self) -> Dict[str, LuaPatternInfo]:
        """Get user-created Lua patterns only."""
        return {
            name: info for name, info in self.lua_patterns.items()
            if str(info.file_path).startswith(str(self.user_patterns_dir))
        }

    # Delegate to YAML engine for backward compatibility
    @property
    def config(self) -> dict:
        """Access YAML config for backward compatibility."""
        return self.yaml_engine.config

    @property
    def config_path(self) -> Path:
        """Access config path for backward compatibility."""
        return self.yaml_engine.config_path

    @property
    def user_config(self) -> dict:
        """Access user config for backward compatibility."""
        return self.yaml_engine.user_config

    @property
    def user_config_path(self) -> Path:
        """Access user config path for backward compatibility."""
        return self.yaml_engine.user_config_path

    @property
    def patterns(self) -> dict:
        """Access loaded patterns for backward compatibility."""
        return self.yaml_engine.patterns

    def extract_digits(self, serial: str) -> str:
        return self.yaml_engine.extract_digits(serial)

    def set_pattern_enabled(self, name: str, enabled: bool):
        """Enable/disable a pattern."""
        if name in self.lua_patterns:
            self.lua_patterns[name].enabled = enabled
        else:
            self.yaml_engine.set_pattern_enabled(name, enabled)

    def add_custom_pattern(self, name: str, defn: dict):
        """Add a custom YAML pattern."""
        self.yaml_engine.add_custom_pattern(name, defn)

    def remove_custom_pattern(self, name: str):
        """Remove a custom YAML pattern."""
        self.yaml_engine.remove_custom_pattern(name)

    def get_custom_patterns(self) -> dict:
        """Get custom YAML patterns."""
        return self.yaml_engine.get_custom_patterns()

    def get_gas_pump_threshold(self) -> float:
        return self.yaml_engine.get_gas_pump_threshold()

    def set_gas_pump_threshold(self, threshold: float):
        self.yaml_engine.set_gas_pump_threshold(threshold)

    def save_config(self):
        """Save YAML user config."""
        self.yaml_engine.save_config()


# Alias for backward compatibility
PatternEngine = PatternEngineV3


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Pattern Engine v3 - Hybrid Test")
    print("=" * 70)

    engine = PatternEngineV3()

    test_serials = [
        ("A88888888B", "SOLID"),
        ("A12344321B", "RADAR"),
        ("A12341234B", "REPEATER"),
        ("A01234567B", "LADDER_UP"),
        ("A76543210B", "LADDER_DOWN"),
        ("A10101010B", "BINARY"),
        ("A11223344B", "DOUBLE_DOUBLE"),
    ]

    print("\nClassification test:")
    for serial, expected in test_serials:
        matches = engine.classify(serial)
        names = [m.name for m in matches]
        status = "?" if expected in names else "?"
        sources = {m.name: m.source for m in matches if m.name in names[:5]}
        print(f"{status} {serial}: {', '.join(names[:5])}")
        print(f"   Sources: {sources}")

    print(f"\nTotal patterns: {len(engine.get_all_patterns())}")
    print(f"  YAML patterns: {len(engine.yaml_engine.get_all_patterns())}")
    print(f"  Lua patterns: {len(engine.lua_patterns)}")

    # Test script execution
    print("\n" + "=" * 70)
    print("Script execution test:")

    test_script = """
    function match(ctx)
        if ctx.digits == string.reverse(ctx.digits) then
            return {
                matched = true,
                highlights = {{positions = {0, 7}, color = "orange"}},
                connectors = {{from = 0, to = 7, color = "orange"}},
                message = "Palindrome detected!"
            }
        end
        return {matched = false}
    end
    """

    result = engine.test_script(test_script, "A12344321B")
    print(f"  Matched: {result.matched}")
    print(f"  Message: {result.message}")
    print(f"  Highlights: {result.highlights}")
    print(f"  Execution time: {result.execution_time_ms:.2f}ms")
