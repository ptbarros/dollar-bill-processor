"""
Pattern Engine v3 - Lua-Only Pattern System

All patterns are defined in Lua scripts under the patterns/ directory.
This is the single source of truth for pattern detection.

Script patterns provide rich visualization with custom highlights and connectors.
"""

import re
from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field

from pattern_sandbox import PatternSandbox, create_context, LuaExecutionResult

if TYPE_CHECKING:
    from settings_manager import SettingsManager


@dataclass
class PatternMatch:
    """Result of a pattern match."""
    name: str
    description: str
    tier: int


@dataclass
class PatternMatchV3(PatternMatch):
    """Extended pattern match with visualization data."""
    highlights: list = field(default_factory=list)
    connectors: list = field(default_factory=list)
    group_boxes: list = field(default_factory=list)  # Boxes spanning multiple digits
    message: str = ""
    source: str = "lua"


@dataclass
class LuaPatternInfo:
    """Information about a loaded Lua pattern."""
    name: str
    description: str
    tier: int
    script: str
    file_path: Path
    library: str = "user"         # Library name (directory under patterns/)
    enabled: bool = True
    display_name: str = ""        # User-friendly name for GUI (falls back to name if empty)
    examples: list = field(default_factory=list)
    odds: str = ""
    price: str = ""
    data_file: str = ""           # Relative path to external data file (CSV/JSON)
    data: Any = None              # Loaded data (list of dicts for CSV, any for JSON)
    data_by_key: dict = None      # Dict keyed by first column (CSV only)
    book_ref: str = ""            # CS- reference number (e.g., "CS-100")
    name_collision: bool = False  # True => this pattern's internal name clashes
                                  # with an already-loaded (winning) pattern, so it
                                  # was set aside inactive instead of overriding it.
    collision_with: str = ""      # Library of the winning pattern it clashed with.
    show_overlay: bool = True     # False => hide from the serial-overlay picker
                                  # (Overlay: none in the header). For patterns
                                  # whose match has nothing useful to draw on the
                                  # digits (STAR, SEAL_SHIFT, low-run, ...).


class PatternEngineV3:
    """
    Lua-only pattern engine.

    Usage:
        engine = PatternEngineV3()
        matches = engine.classify("A12344321B")
        highlights = engine.get_digit_highlights("A12344321B", ["RADAR"])
    """

    def __init__(self, patterns_dir: Path = None):
        """
        Initialize the pattern engine.

        Args:
            patterns_dir: Path to patterns/ directory (default: auto-detect)
        """
        # Base directory (repo root when running from source, bundle data dir
        # when frozen by PyInstaller).
        from resource_path import app_base
        base_dir = app_base()

        # Settings manager (lazy loaded)
        self._settings: Optional['SettingsManager'] = None

        # Set up patterns directory
        if patterns_dir is None:
            self.patterns_dir = base_dir / "patterns"
        else:
            self.patterns_dir = Path(patterns_dir)

        # Initialize Lua sandbox
        self.sandbox = PatternSandbox()
        self._load_helpers()

        # User patterns must be writable. When frozen the bundle is read-only,
        # so user patterns live in the writable per-user data dir instead of the
        # (bundled) patterns/ tree. Set before loading so they get picked up.
        import sys
        if getattr(sys, "frozen", False):
            from resource_path import user_data_dir
            self.user_patterns_dir = user_data_dir() / "patterns" / "user"
            try:
                self.user_patterns_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        else:
            self.user_patterns_dir = self.patterns_dir / "user"

        # Load Lua patterns
        self.lua_patterns: Dict[str, LuaPatternInfo] = {}
        # Patterns whose internal name collided with an already-loaded one (the
        # first-loaded wins, so built-ins beat a user duplicate). Kept here,
        # inactive, so the UI can flag them for the user to rename/move/delete.
        self.shadowed_patterns: List[LuaPatternInfo] = []
        self._load_lua_patterns()

    @property
    def settings(self) -> Optional['SettingsManager']:
        """Lazy-load settings manager."""
        if self._settings is None:
            try:
                from settings_manager import get_settings
                self._settings = get_settings()
            except ImportError:
                self._settings = None
        return self._settings

    def _load_helpers(self):
        """Load helper functions into sandbox."""
        helpers_path = self.patterns_dir / "lib" / "helpers.lua"
        if helpers_path.exists():
            with open(helpers_path, 'r', encoding='utf-8') as f:
                helpers_code = f.read()
            self.sandbox.load_helpers(helpers_code)

    def _load_shipped_enabled(self):
        """Internal pattern names that make up the shipped default set (whitelist).

        Read from patterns/shipped_enabled.json (the curated core from Ed's review).
        When present, any pattern NOT in this list ships DISABLED by default; the
        .lua files stay in place so they remain re-enableable and importable. A
        user's explicit on/off in settings always overrides this default. If the
        file is missing/empty, no whitelist is applied (every pattern defaults on).
        """
        import json
        path = self.patterns_dir / "shipped_enabled.json"
        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return set(json.load(f))
        except Exception:
            pass
        return set()

    def _load_lua_patterns(self):
        """Load all Lua pattern scripts from all library directories."""
        self.lua_patterns.clear()
        self.shadowed_patterns.clear()

        # Skip these directories (not pattern libraries)
        skip_dirs = {'lib', 'data', '__pycache__'}

        # Curated shipped set (whitelist). Patterns not in it ship disabled by
        # default; the files remain for re-enable/import.
        self._shipped_enabled = self._load_shipped_enabled()

        # Scan all subdirectories under patterns/
        if self.patterns_dir.exists():
            for subdir in sorted(self.patterns_dir.iterdir()):
                if not subdir.is_dir() or subdir.name in skip_dirs:
                    continue

                library_name = subdir.name

                for lua_file in subdir.glob("*.lua"):
                    self._load_lua_pattern(lua_file, library=library_name)

        # Also load patterns from the writable user-data tree when it's separate
        # from the bundled tree (frozen build): the flat "user" folder (personal
        # saved patterns) PLUS any NAMED add-on libraries imported from a bundle
        # (e.g. "Green Guide"), each scanned as its own library so it shows as its
        # own group and can be removed as a group. Loaded after the bundle, so a
        # user-side pattern overrides a bundled one of the same internal name.
        user_root = self.user_libraries_root()
        if user_root and user_root != self.patterns_dir and user_root.exists():
            for subdir in sorted(user_root.iterdir()):
                if not subdir.is_dir() or subdir.name in skip_dirs:
                    continue
                for lua_file in subdir.glob("*.lua"):
                    self._load_lua_pattern(lua_file, library=subdir.name)

    def user_libraries_root(self) -> Optional[Path]:
        """Writable dir holding user-side libraries — the flat 'user' folder plus
        named add-on libraries imported from bundles. In a frozen build this is
        user_data_dir/patterns (separate from the read-only bundle); from source
        it's the repo patterns/ tree (already scanned above), returned so import
        still has a destination."""
        if self.user_patterns_dir != self.patterns_dir / "user":
            return self.user_patterns_dir.parent
        return self.patterns_dir

    def removable_libraries(self) -> List[str]:
        """Named add-on libraries the user may remove (imported bundles): folders
        under the user-libraries root, excluding 'user', helper dirs, and any
        bundled (shipped, read-only) library."""
        root = self.user_libraries_root()
        if not root or not root.exists():
            return []
        skip = {'lib', 'data', '__pycache__', 'user'}
        bundled = {p.name for p in self.patterns_dir.iterdir()
                   if p.is_dir()} if self.patterns_dir.exists() else set()
        return sorted(d.name for d in root.iterdir()
                      if d.is_dir() and d.name not in skip and d.name not in bundled)

    def remove_library(self, library_name: str) -> bool:
        """Delete an imported add-on library folder, then reload. Only a user-side,
        non-bundled library can be removed (guards against deleting shipped ones).
        Returns True if it was removed."""
        if library_name not in self.removable_libraries():
            return False
        import shutil
        try:
            shutil.rmtree(self.user_libraries_root() / library_name)
        except Exception:
            return False
        self.reload()
        return True

    # Curated libraries that ship with the app -- their patterns are read-only to
    # the move feature (a user can't move them out, and nothing can move into
    # them). Everything else (the flat 'user' folder + imported/created add-on
    # libraries) is user-side and movable.
    SHIPPED_LIBRARIES = frozenset({'core', 'Nicks', 'The Green Guide', 'Essentials'})
    _NON_LIBRARY_DIRS = frozenset({'lib', 'data', '__pycache__'})

    def movable_pattern_libraries(self) -> list:
        """User-side libraries a pattern may be moved between: the flat 'user'
        folder plus imported/created add-on libraries. Never core or any shipped
        library."""
        libs = {"user"}
        root = self.user_libraries_root()
        if root and root.exists():
            for d in root.iterdir():
                if (d.is_dir() and d.name not in self._NON_LIBRARY_DIRS
                        and d.name not in self.SHIPPED_LIBRARIES):
                    libs.add(d.name)
        return sorted(libs)

    def move_pattern(self, name: str, target_library: str) -> tuple:
        """Move a user-created pattern (and a copy of its data file, if any) into
        another user-side library. Refuses to move a built-in/shipped pattern or to
        move into core/a shipped library. Returns (ok, message)."""
        # Only ACTIVE patterns are movable. A shadowed name-collision loser keeps
        # the same internal name, so moving it wouldn't stop the clash -- those are
        # resolved by delete/rename via the collision flag instead.
        info = self.lua_patterns.get(name)
        if info is None:
            return False, f"Pattern '{name}' was not found."

        # Source must be a user-side library (never core or a shipped library).
        if info.library in self.SHIPPED_LIBRARIES:
            return False, (f"'{name}' is in the built-in '{info.library}' library "
                           "and can't be moved.")

        # Destination must not be core or a shipped/built-in library.
        if target_library == "core" or target_library in self.SHIPPED_LIBRARIES:
            return False, f"Patterns can't be moved into the built-in '{target_library}' library."
        if target_library in self._NON_LIBRARY_DIRS:
            return False, f"'{target_library}' is not a valid library name."

        if target_library == info.library:
            return False, f"'{name}' is already in '{target_library}'."

        target_dir = (self.user_patterns_dir if target_library == "user"
                      else (self.user_libraries_root() or self.patterns_dir) / target_library)
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return False, f"Could not create the destination library: {e}"

        src = Path(info.file_path)
        dest = target_dir / src.name
        if dest.exists():
            return False, f"A file named {src.name} already exists in '{target_library}'."

        import shutil
        try:
            # Copy the data file first (leave the original in place -- other
            # patterns in the source library may share it), then move the .lua.
            if info.data_file:
                data_src = src.parent / Path(info.data_file).name
                if data_src.exists():
                    data_dest = target_dir / data_src.name
                    if not data_dest.exists():
                        shutil.copy2(str(data_src), str(data_dest))
            shutil.move(str(src), str(dest))
        except Exception as e:
            return False, f"Move failed: {e}"

        self.reload()
        return True, f"Moved '{name}' to '{target_library}'."

    def _load_lua_pattern(self, file_path: Path, library: str = "user"):
        """Load a single Lua pattern from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                script = f.read()

            # Parse metadata from header comment
            metadata = self._parse_lua_metadata(script)

            # Use Pattern: field from header, or filename as fallback
            name = metadata.get('pattern', file_path.stem.upper())

            # Validate syntax
            valid, error = self.sandbox.validate_syntax(script)
            if not valid:
                print(f"Warning: Syntax error in {file_path}: {error}")
                return

            # Determine enabled state from settings (pattern state overrides library state)
            # Default: enabled unless explicitly disabled in settings
            enabled = True
            if self.settings:
                # Check library enabled state first. The bundled "Essentials"
                # library ships DISABLED by default (it duplicates concepts that
                # also live in core/Nicks/Green Guide, so leaving it on would
                # double-fire) — the user enables it deliberately to try it.
                lib_default = library != "Essentials"
                lib_enabled = self.settings.get_library_enabled(library, default=lib_default)
                # Curated shipped set (Ed review): when a whitelist is present, a
                # pattern outside it defaults off. Library state still gates it, and
                # the user's explicit choice overrides both.
                whitelist = getattr(self, '_shipped_enabled', set())
                if whitelist:
                    ship_default = lib_enabled and (name in whitelist)
                else:
                    ship_default = lib_enabled
                # Check pattern-specific state (overrides library + ship default)
                pattern_enabled = self.settings.get_pattern_enabled(name, default=ship_default)
                enabled = pattern_enabled

            # Create pattern info
            info = LuaPatternInfo(
                name=name,
                description=metadata.get('description', ''),
                tier=int(metadata.get('tier', 10)),
                script=script,
                file_path=file_path,
                library=library,
                enabled=enabled,
                display_name=metadata.get('displayname', ''),
                examples=metadata.get('examples', []),
                odds=metadata.get('odds', ''),
                price=metadata.get('price', ''),
                book_ref=metadata.get('bookref', ''),
                # "Overlay: none/off/false/no" hides the pattern from the serial
                # overlay picker (still matches + shows in results). Default shown.
                show_overlay=str(metadata.get('overlay', '')).strip().lower()
                not in ('none', 'off', 'false', 'no', '0'),
            )

            # Load external data file if specified
            data_file = metadata.get('datafile', '')
            if data_file:
                data, data_by_key = self._load_data_file(data_file, file_path)
                info.data_file = data_file
                info.data = data
                info.data_by_key = data_by_key

            # Name-collision handling: the first-loaded pattern wins. Library scan
            # order loads shipped libraries (core, Nicks, ...) before user-side
            # ones, so a built-in beats a user duplicate. Instead of the old
            # behavior (the later one silently overwrote/hid the earlier), set the
            # loser aside, inactive and flagged, so the UI can surface it and the
            # user can rename/move/delete it.
            existing = self.lua_patterns.get(name)
            if existing is not None:
                info.enabled = False
                info.name_collision = True
                info.collision_with = existing.library
                self.shadowed_patterns.append(info)
                return

            self.lua_patterns[name] = info

        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")

    def get_name_collisions(self) -> list:
        """User-side patterns shadowed by a same-named winning pattern (built-in
        or earlier-loaded). Each is inactive; the UI flags these so the user can
        rename/delete them. Shipped-vs-shipped clashes are omitted (nothing the
        user can act on)."""
        return [info for info in self.shadowed_patterns
                if info.library not in self.SHIPPED_LIBRARIES]

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
        """Reload all Lua patterns."""
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

        # Run Lua patterns
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
                        tier=self._effective_tier(name, info.tier),
                        highlights=result.highlights,
                        connectors=result.connectors,
                        group_boxes=result.group_boxes,
                        message=result.message,
                        source="lua"
                    ))
            except Exception as e:
                # Log but don't fail on individual pattern errors
                pass

        # Sort by tier
        matches.sort(key=lambda m: (m.tier, m.name))
        return matches

    def classify_simple(self, serial: str, metadata: dict = None) -> List[str]:
        """Return just pattern names."""
        return [m.name for m in self.classify(serial, metadata)]

    def classify_reference(self, serial: str, allowed, metadata: dict = None) -> List[str]:
        """Classify against a specific set of pattern names (a reference/'originals'
        selection), ignoring the current enabled state. Temporarily enables only
        the names in `allowed`, then restores. No settings are written."""
        allowed = set(allowed)
        snapshot = {name: p.enabled for name, p in self.lua_patterns.items()}
        try:
            for name, p in self.lua_patterns.items():
                p.enabled = name in allowed
            return self.classify_simple(serial, metadata)
        finally:
            for name, p in self.lua_patterns.items():
                p.enabled = snapshot.get(name, p.enabled)

    def classify_full(self, serial: str, metadata: dict = None) -> List[str]:
        """Classify against the ENTIRE installed library, ignoring the current
        enabled/disabled selection. Temporarily enables every pattern in memory
        (no settings are written) and restores the prior state afterward. Used by
        the Core-vs-Full coverage check to see what a lean set would miss."""
        snapshot = {name: p.enabled for name, p in self.lua_patterns.items()}
        try:
            for p in self.lua_patterns.values():
                p.enabled = True
            return self.classify_simple(serial, metadata)
        finally:
            for name, p in self.lua_patterns.items():
                p.enabled = snapshot.get(name, p.enabled)

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
                # Draw the overlay for any explicitly-requested pattern
                # regardless of its enabled state: callers pass a specific
                # pattern the bill already matched (the selected overlay) or one
                # being previewed, and want its highlights even if the set it
                # belongs to is currently disabled — e.g. after an A/B library
                # toggle, a bill's stored classification can reference a
                # now-disabled pattern. result.matched is the real gate.
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

        return {'highlights': all_highlights, 'connectors': all_connectors, 'group_boxes': all_group_boxes}

    def _merge_lua_highlights(self, all_highlights: list, lua_highlights: list, pattern_name: str):
        """Merge Lua script highlights into the per-position format."""
        for h in lua_highlights:
            positions = h.get('positions', [])
            color = h.get('color', 'gray')
            label = h.get('label', '')
            # 'box' (default), 'x' (X only, no box), or 'boxed_x' (box + X). Used to
            # mark an EXCLUDED/leftover digit with an X instead of a box, which
            # can't be mistaken for a match the way a muted box can.
            style = h.get('style', '')

            for pos in positions:
                if 0 <= pos < 8:
                    all_highlights[pos]['highlights'].append({
                        'pattern': pattern_name,
                        'color': color,
                        'style': style,
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
            # Pattern not found
            return LuaExecutionResult(
                success=False,
                matched=False,
                message=f"Pattern '{pattern_name}' not found"
            )

    def test_script(self, script: str, serial: str, metadata: dict = None, debug: bool = False) -> LuaExecutionResult:
        """
        Test a pattern script without saving it.

        Args:
            script: Lua script code
            serial: Serial number to test
            metadata: Optional metadata
            debug: If True, enable log() function to collect debug messages

        Returns:
            LuaExecutionResult with execution details and debug_log
        """
        ctx = create_context(serial, metadata)
        return self.sandbox.execute(script, ctx, debug=debug)

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

        # Strip any existing header block(s) from the script to avoid duplicates
        import re
        # Remove all --[[ ... --]] blocks at the start of the script
        cleaned_script = script.strip()
        while cleaned_script.startswith('--[['):
            # Find the closing --]]
            end_pos = cleaned_script.find('--]]')
            if end_pos != -1:
                cleaned_script = cleaned_script[end_pos + 4:].strip()
            else:
                break  # Malformed header, stop trying

        # Combine header and cleaned script
        full_script = f"{header}\n\n{cleaned_script}"

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(full_script)

            # Reload to pick up the new pattern
            self._load_lua_pattern(file_path, library=library)
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

    def delete_collision(self, file_path) -> bool:
        """Delete a shadowed (name-collision) user-side pattern file, then reload.
        For safety only a file currently in the user-side collision list can be
        removed this way (used to resolve a duplicate that hides a built-in)."""
        fp = Path(file_path)
        if str(fp) not in {str(c.file_path) for c in self.get_name_collisions()}:
            return False
        try:
            fp.unlink()
        except Exception as e:
            print(f"Error deleting duplicate pattern: {e}")
            return False
        self.reload()
        return True

    # Short words that read better lowercased mid-name (else the "short uppercase
    # acronym" rule below keeps them uppercase, giving "Sum 61 OR 11").
    _JOINER_WORDS = {'OF', 'IN', 'OR', 'AND', 'FOR', 'THE', 'A', 'TO', 'ON', 'AT', 'BY'}

    def _make_friendly_name(self, name: str) -> str:
        """Convert PATTERN_NAME to 'Pattern Name' for display."""
        # Split on underscores and capitalize each word
        words = name.split('_')
        # Keep numbers/acronyms as-is, title-case others
        result = []
        for i, word in enumerate(words):
            if i > 0 and word.upper() in self._JOINER_WORDS:
                # Lowercase joiner words mid-name (of/in/or/and), not the first word
                result.append(word.lower())
            elif word.isdigit() or (len(word) <= 2 and word.isupper()):
                # Keep short acronyms and numbers as-is (e.g., "6M", "12M")
                result.append(word)
            else:
                result.append(word.capitalize())
        return ' '.join(result)

    def _effective_tier(self, name: str, base_tier: int) -> int:
        """The pattern's tier with the user's settings-layer override applied.

        Mirrors the label override: a user can reorder a pattern (e.g. its
        placement in the overlay cycle, which sorts lowest tier first) by its
        perceived value without editing the .lua. Falls back to the .lua tier."""
        if self.settings:
            override = self.settings.get_pattern_tier(name)
            if override:
                return int(override)
        return base_tier

    def get_pattern_info(self, name: str) -> Optional[dict]:
        """Get info about a pattern."""
        if name in self.lua_patterns:
            info = self.lua_patterns[name]
            # User label override (settings layer) wins over the .lua DisplayName,
            # so a read-only core pattern can be relabeled without editing it.
            label_override = self.settings.get_pattern_label(name) if self.settings else ""
            result = {
                'name': info.name,
                'display_name': label_override or info.display_name or self._make_friendly_name(info.name),
                'library': info.library,
                'description': info.description,
                'tier': self._effective_tier(name, info.tier),
                'enabled': info.enabled,
                'source': 'lua',
                'examples': info.examples,
                'odds': info.odds,
                'price': info.price,
                'price_range': info.price,  # Alias for backward compatibility
                'script': info.script
            }
            # Include data file info if present
            if info.data_file:
                result['data_file'] = info.data_file
                result['data_loaded'] = info.data is not None
            return result
        return None

    def get_all_patterns(self) -> dict:
        """Get all patterns."""
        patterns = {}

        for name, info in self.lua_patterns.items():
            patterns[name] = {
                'name': info.name,
                'description': info.description,
                'tier': self._effective_tier(name, info.tier),
                'enabled': info.enabled,
                'source': 'lua',
                'examples': info.examples,
                'odds': info.odds,
                'price': info.price,
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

    def extract_digits(self, serial: str) -> str:
        """Extract numeric portion of serial."""
        return ''.join(c for c in serial if c.isdigit())

    def set_pattern_enabled(self, name: str, enabled: bool):
        """Enable/disable a pattern."""
        if name in self.lua_patterns:
            self.lua_patterns[name].enabled = enabled
            # Persist to settings
            if self.settings:
                self.settings.set_pattern_enabled(name, enabled)

    def sync_enabled_from_settings(self):
        """Refresh every loaded pattern's `enabled` flag from the (shared) settings,
        WITHOUT re-parsing the Lua. Other engine instances (Pattern Manager, the
        results list, the processor) each hold their own in-memory flags; when one
        changes enable/disable, the others go stale until restart. Call this on an
        engine before it's used (Strap Check, Serial Lookup) so it reflects the
        current settings. Mirrors the enabled logic in _load_patterns."""
        if not self.settings:
            return
        whitelist = getattr(self, '_shipped_enabled', set())
        for name, info in self.lua_patterns.items():
            lib_default = info.library != "Essentials"
            lib_enabled = self.settings.get_library_enabled(info.library, default=lib_default)
            if whitelist:
                ship_default = lib_enabled and (name in whitelist)
            else:
                ship_default = lib_enabled
            info.enabled = self.settings.get_pattern_enabled(name, default=ship_default)

    def clear_pattern_enabled(self, name: str):
        """Clear explicit pattern state, reverting to library default.

        This is used when toggling a library checkbox - we want the library
        state to control patterns rather than individual overrides.
        """
        if name in self.lua_patterns:
            info = self.lua_patterns[name]
            # Clear from settings (removes explicit override)
            if self.settings:
                self.settings.clear_pattern_enabled(name)
                # Recalculate enabled state from library default
                lib_enabled = self.settings.get_library_enabled(info.library, default=True)
                info.enabled = lib_enabled

    def get_gas_pump_threshold(self) -> float:
        """Get the GAS_PUMP baseline_variance_min threshold.

        Checks SettingsManager first, then defaults to 3.5.
        """
        if self.settings:
            return self.settings.get_gas_pump_threshold(default=3.5)
        return 3.5

    def set_gas_pump_threshold(self, threshold: float):
        """Set the GAS_PUMP baseline_variance_min threshold and save.

        Updates the SettingsManager and saves to file.
        """
        if self.settings:
            self.settings.set_gas_pump_threshold(threshold)
            self.settings.save()


# Alias for backward compatibility
PatternEngine = PatternEngineV3


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Pattern Engine v3 - Lua-Only Test")
    print("=" * 70)

    engine = PatternEngineV3()

    test_serials = [
        ("A88888888B", "solid/8-of-a-kind"),  # All 8s
        ("A12344321B", "radar/palindrome"),  # Palindrome
        ("A12341234B", "repeater"),  # ABCDABCD
        ("A01234567B", "ladder"),  # Full 8-digit ladder
        ("A76543210B", "ladder"),  # Descending
        ("A00001111B", "binary"),  # Only 0s and 1s
        ("A11223344B", "doubles ladder"),  # Pairs forming ladder
    ]

    print("\nClassification test (patterns must match):")
    for serial, desc in test_serials:
        matches = engine.classify(serial)
        names = [m.name for m in matches]
        status = "✓" if names else "✗"
        print(f"{status} {serial} ({desc}): {', '.join(names[:5])}")

    print(f"\nTotal patterns: {len(engine.get_all_patterns())}")
    print(f"  Lua patterns: {len(engine.lua_patterns)}")

    # Show library breakdown
    libs = {}
    for name, info in engine.lua_patterns.items():
        lib = info.library
        if lib not in libs:
            libs[lib] = {'enabled': 0, 'disabled': 0}
        if info.enabled:
            libs[lib]['enabled'] += 1
        else:
            libs[lib]['disabled'] += 1

    print("\n  By library:")
    for lib, counts in sorted(libs.items()):
        print(f"    {lib}: {counts['enabled']} enabled, {counts['disabled']} disabled")

    # Test pattern info
    print("\n" + "=" * 70)
    print("Pattern info test (RADAR):")
    info = engine.get_pattern_info("RADAR")
    if info:
        print(f"  Description: {info.get('description')}")
        print(f"  Tier: {info.get('tier')}")
        print(f"  Odds: {info.get('odds')}")
        print(f"  Price: {info.get('price')}")
    else:
        print("  Pattern not found!")

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
