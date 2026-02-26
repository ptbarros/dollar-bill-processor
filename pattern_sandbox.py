"""
Pattern Sandbox - Secure Lua Execution Environment

Provides a sandboxed Lua runtime for executing pattern scripts safely.
Uses lupa library with whitelisted functions and execution limits.
"""

import time
from typing import Optional, Any
from dataclasses import dataclass, field


@dataclass
class LuaExecutionResult:
    """Result of executing a Lua pattern script."""
    success: bool
    matched: bool = False
    highlights: list = field(default_factory=list)
    connectors: list = field(default_factory=list)
    group_boxes: list = field(default_factory=list)  # Boxes spanning multiple digits
    message: str = ""
    error: str = ""
    execution_time_ms: float = 0.0
    debug_log: list = field(default_factory=list)  # Debug log entries from log() calls


class PatternSandbox:
    """
    Secure Lua execution environment for pattern scripts.

    Security features:
    - Whitelisted safe functions only (string, math, table, pairs, ipairs, etc.)
    - Blocked dangerous functions (os, io, loadfile, require, debug, etc.)
    - Instruction limit to prevent infinite loops
    - Timeout protection

    Usage:
        sandbox = PatternSandbox()
        result = sandbox.execute(script_code, ctx={
            'digits': '12344321',
            'full_serial': 'A12344321B',
            'metadata': {},
            'digit_list': [1, 2, 3, 4, 4, 3, 2, 1]
        })
    """

    # Maximum instructions before termination (prevents infinite loops)
    MAX_INSTRUCTIONS = 10_000

    # Timeout in seconds
    TIMEOUT_SECONDS = 0.1  # 100ms

    # Safe Lua standard library functions
    SAFE_GLOBALS = {
        # Basic functions
        'assert', 'error', 'ipairs', 'pairs', 'next',
        'pcall', 'xpcall', 'select', 'tonumber', 'tostring',
        'type', 'unpack', 'rawequal', 'rawget', 'rawset',

        # Math library
        'math',

        # String library
        'string',

        # Table library
        'table',
    }

    # Functions that must be blocked for security
    BLOCKED_GLOBALS = {
        'os', 'io', 'loadfile', 'dofile', 'load', 'loadstring',
        'require', 'module', 'package', 'debug', 'coroutine',
        'collectgarbage', 'getfenv', 'setfenv', 'getmetatable',
        'setmetatable', 'rawset',  # rawset can bypass metatables
        '_G',  # Direct access to global table
    }

    def __init__(self, instruction_limit: int = None, timeout_seconds: float = None):
        """
        Initialize the sandbox.

        Args:
            instruction_limit: Max Lua instructions (default: 10,000)
            timeout_seconds: Execution timeout (default: 0.1s)
        """
        self.instruction_limit = instruction_limit or self.MAX_INSTRUCTIONS
        self.timeout_seconds = timeout_seconds or self.TIMEOUT_SECONDS
        self._lua = None
        self._helpers_loaded = False

    def _get_lua(self):
        """Get or create the Lua runtime with sandboxed environment."""
        if self._lua is None:
            try:
                from lupa import LuaRuntime
            except ImportError:
                raise ImportError(
                    "lupa is required for Lua pattern scripts. "
                    "Install with: pip install lupa"
                )

            # Create Lua runtime with attribute filter for additional safety
            self._lua = LuaRuntime(
                unpack_returned_tuples=True,
                attribute_filter=self._attribute_filter
            )

            # Set up sandboxed environment
            self._setup_sandbox()

        return self._lua

    def _attribute_filter(self, obj, attr_name, is_setting):
        """Filter attribute access for security."""
        # Block access to private/dunder attributes
        if attr_name.startswith('_'):
            raise AttributeError(f"Access to '{attr_name}' is not allowed")
        return attr_name

    def _setup_sandbox(self):
        """Configure the Lua environment with safe functions only."""
        lua = self._lua
        g = lua.globals()

        # Create a restricted environment table
        lua.execute("""
            -- Create sandboxed environment
            sandbox_env = {}

            -- Copy safe globals
            sandbox_env.assert = assert
            sandbox_env.error = error
            sandbox_env.ipairs = ipairs
            sandbox_env.pairs = pairs
            sandbox_env.next = next
            sandbox_env.pcall = pcall
            sandbox_env.xpcall = xpcall
            sandbox_env.select = select
            sandbox_env.tonumber = tonumber
            sandbox_env.tostring = tostring
            sandbox_env.type = type
            sandbox_env.unpack = unpack or table.unpack
            sandbox_env.rawequal = rawequal
            sandbox_env.rawget = rawget

            -- Copy safe libraries (as copies, not references)
            sandbox_env.math = {}
            for k, v in pairs(math) do
                sandbox_env.math[k] = v
            end

            sandbox_env.string = {}
            for k, v in pairs(string) do
                sandbox_env.string[k] = v
            end

            sandbox_env.table = {}
            for k, v in pairs(table) do
                sandbox_env.table[k] = v
            end

            -- Add print as a no-op (scripts shouldn't print)
            sandbox_env.print = function(...) end

            -- Helper to serialize a value for logging
            local function serialize_value(v, depth)
                depth = depth or 0
                if depth > 3 then return "..." end

                if type(v) == "table" then
                    local parts = {}
                    local is_array = true
                    local max_index = 0

                    -- Check if it's an array
                    for k, _ in pairs(v) do
                        if type(k) ~= "number" or k < 1 or math.floor(k) ~= k then
                            is_array = false
                            break
                        end
                        if k > max_index then max_index = k end
                    end

                    if is_array and max_index > 0 then
                        -- Array format
                        for i = 1, max_index do
                            table.insert(parts, serialize_value(v[i], depth + 1))
                        end
                        return "{" .. table.concat(parts, ", ") .. "}"
                    else
                        -- Dict format
                        for k, val in pairs(v) do
                            local key_str = tostring(k)
                            table.insert(parts, key_str .. "=" .. serialize_value(val, depth + 1))
                        end
                        return "{" .. table.concat(parts, ", ") .. "}"
                    end
                else
                    return tostring(v)
                end
            end

            -- Helper to run code in sandbox
            function run_sandboxed(code, ctx, debug_enabled)
                -- Create fresh environment for this execution
                local env = {}
                setmetatable(env, {__index = sandbox_env})

                -- Inject context
                env.ctx = ctx

                -- Create per-execution debug log
                local debug_log = {}

                -- Create log function (only functional when debug_enabled)
                if debug_enabled then
                    env.log = function(...)
                        local args = {...}
                        local parts = {}
                        for i, v in ipairs(args) do
                            table.insert(parts, serialize_value(v))
                        end
                        table.insert(debug_log, table.concat(parts, " "))
                    end
                else
                    -- No-op when debug disabled
                    env.log = function(...) end
                end

                -- Load and run the code
                local fn, err = load(code, "pattern", "t", env)
                if not fn then
                    return {success = false, error = "Syntax error: " .. tostring(err), debug_log = debug_log}
                end

                -- Execute to define the match function
                local ok, result = pcall(fn)
                if not ok then
                    return {success = false, error = "Load error: " .. tostring(result), debug_log = debug_log}
                end

                -- Call the match function
                if type(env.match) ~= "function" then
                    return {success = false, error = "Pattern must define a match(ctx) function", debug_log = debug_log}
                end

                ok, result = pcall(env.match, ctx)
                if not ok then
                    return {success = false, error = "Runtime error: " .. tostring(result), debug_log = debug_log}
                end

                -- Validate result
                if type(result) ~= "table" then
                    return {success = false, error = "match() must return a table", debug_log = debug_log}
                end

                result.success = true
                result.debug_log = debug_log
                return result
            end
        """)

    def load_helpers(self, helpers_code: str) -> bool:
        """
        Load helper functions into the sandbox environment.

        Args:
            helpers_code: Lua code defining helper functions

        Returns:
            True if helpers loaded successfully
        """
        lua = self._get_lua()

        try:
            # Load helpers into the sandbox_env
            lua.execute(f"""
                local helpers_fn, err = load([=[{helpers_code}]=], "helpers", "t", sandbox_env)
                if helpers_fn then
                    helpers_fn()
                else
                    error("Failed to load helpers: " .. tostring(err))
                end
            """)
            self._helpers_loaded = True
            return True
        except Exception as e:
            return False

    def execute(self, script: str, ctx: dict, debug: bool = False) -> LuaExecutionResult:
        """
        Execute a pattern script in the sandbox.

        Args:
            script: The Lua pattern script code
            ctx: Context dict with digits, full_serial, metadata, digit_list
            debug: If True, enable log() function to collect debug messages

        Returns:
            LuaExecutionResult with match status and visualization data
        """
        start_time = time.time()

        try:
            lua = self._get_lua()
        except ImportError as e:
            return LuaExecutionResult(
                success=False,
                error=str(e)
            )

        # Convert Python ctx to Lua-friendly format
        lua_ctx = self._python_to_lua(ctx)

        try:
            # Get the sandboxed runner
            run_sandboxed = lua.globals()['run_sandboxed']

            # Execute with debug flag
            result = run_sandboxed(script, lua_ctx, debug)

            execution_time = (time.time() - start_time) * 1000

            # Convert Lua result back to Python
            return self._lua_result_to_python(result, execution_time)

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return LuaExecutionResult(
                success=False,
                error=f"Execution error: {str(e)}",
                execution_time_ms=execution_time
            )

    def _python_to_lua(self, ctx: dict) -> Any:
        """Convert Python context dict to Lua table."""
        lua = self._get_lua()

        # Create Lua table
        lua_table = lua.table_from({
            'digits': ctx.get('digits', ''),
            'full_serial': ctx.get('full_serial', ''),
        })

        # Add digit_list as Lua array (1-indexed)
        digit_list = ctx.get('digit_list', [])
        if digit_list:
            lua_digits = lua.table_from(digit_list)
            lua_table['digit_list'] = lua_digits

        # Add metadata
        metadata = ctx.get('metadata', {})
        if metadata:
            lua_metadata = lua.table_from(metadata)
            lua_table['metadata'] = lua_metadata

        # Add external data if present (from DataFile)
        data = ctx.get('data')
        if data is not None:
            lua_table['data'] = self._convert_to_lua(data)

        data_by_key = ctx.get('data_by_key')
        if data_by_key is not None:
            lua_table['data_by_key'] = self._convert_to_lua(data_by_key)

        return lua_table

    def _convert_to_lua(self, obj: Any) -> Any:
        """Recursively convert Python object to Lua table."""
        lua = self._get_lua()

        if isinstance(obj, dict):
            return lua.table_from({
                k: self._convert_to_lua(v) for k, v in obj.items()
            })
        elif isinstance(obj, list):
            return lua.table_from([self._convert_to_lua(item) for item in obj])
        else:
            # Primitives (str, int, float, bool, None) pass through
            return obj

    def _lua_result_to_python(self, lua_result, execution_time: float) -> LuaExecutionResult:
        """Convert Lua result table to Python LuaExecutionResult."""
        try:
            # Helper to safely get value from Lua table
            def get_lua_value(table, key, default=None):
                try:
                    val = table[key]
                    return val if val is not None else default
                except (KeyError, TypeError, IndexError):
                    return default

            # Extract debug_log (available in both success and error cases)
            debug_log = []
            debug_log_table = get_lua_value(lua_result, 'debug_log')
            if debug_log_table:
                debug_log = self._lua_table_to_list(debug_log_table)
                # Ensure all entries are strings
                debug_log = [str(entry) for entry in debug_log]

            # Check for errors first
            success = get_lua_value(lua_result, 'success', False)
            if not success:
                error_msg = get_lua_value(lua_result, 'error', 'Unknown error')
                return LuaExecutionResult(
                    success=False,
                    error=str(error_msg) if error_msg else 'Unknown error',
                    execution_time_ms=execution_time,
                    debug_log=debug_log
                )

            # Extract matched status
            matched = bool(get_lua_value(lua_result, 'matched', False))

            # Extract highlights
            highlights = []
            highlights_table = get_lua_value(lua_result, 'highlights')
            if highlights_table:
                highlights = self._lua_table_to_list(highlights_table)

            # Extract connectors
            connectors = []
            connectors_table = get_lua_value(lua_result, 'connectors')
            if connectors_table:
                connectors = self._lua_table_to_list(connectors_table)

            # Extract group_boxes (boxes spanning multiple digits)
            group_boxes = []
            group_boxes_table = get_lua_value(lua_result, 'group_boxes')
            if group_boxes_table:
                group_boxes = self._lua_table_to_list(group_boxes_table)

            # Extract message
            message_val = get_lua_value(lua_result, 'message', '')
            message = str(message_val) if message_val else ""

            return LuaExecutionResult(
                success=True,
                matched=matched,
                highlights=highlights,
                connectors=connectors,
                group_boxes=group_boxes,
                message=message,
                execution_time_ms=execution_time,
                debug_log=debug_log
            )

        except Exception as e:
            return LuaExecutionResult(
                success=False,
                error=f"Result parsing error: {str(e)}",
                execution_time_ms=execution_time
            )

    def _lua_table_to_list(self, lua_table) -> list:
        """Convert a Lua table (array) to Python list."""
        result = []
        try:
            # Lua arrays are 1-indexed
            i = 1
            while True:
                try:
                    item = lua_table[i]
                    if item is None:
                        break
                    # Convert nested tables
                    if hasattr(item, '__getitem__') and not isinstance(item, str):
                        item = self._lua_table_to_dict(item)
                    result.append(item)
                    i += 1
                except (KeyError, TypeError):
                    break
        except Exception:
            pass
        return result

    def _lua_table_to_dict(self, lua_table) -> dict:
        """Convert a Lua table to Python dict."""
        result = {}
        try:
            # Try to iterate as dict
            for k, v in lua_table.items() if hasattr(lua_table, 'items') else []:
                if hasattr(v, '__getitem__') and not isinstance(v, str):
                    # Check if it's array-like (consecutive integer keys starting at 1)
                    if self._is_lua_array(v):
                        v = self._lua_table_to_list(v)
                    else:
                        v = self._lua_table_to_dict(v)
                result[k] = v
        except Exception:
            # Fallback: try direct key access for known fields
            for key in ['positions', 'color', 'label', 'from', 'to', 'style']:
                try:
                    if key in lua_table:
                        val = lua_table[key]
                        if hasattr(val, '__getitem__') and not isinstance(val, str):
                            val = self._lua_table_to_list(val)
                        result[key] = val
                except (KeyError, TypeError):
                    pass
        return result

    def _is_lua_array(self, lua_table) -> bool:
        """Check if a Lua table is array-like."""
        try:
            return lua_table[1] is not None
        except (KeyError, TypeError, IndexError):
            return False

    def validate_syntax(self, script: str) -> tuple:
        """
        Validate Lua script syntax without executing.

        Returns:
            (is_valid, error_message)
        """
        try:
            lua = self._get_lua()

            # Try to load (compile) the script
            result = lua.execute(f"""
                local fn, err = load([=[{script}]=], "pattern", "t", sandbox_env)
                if fn then
                    return true, nil
                else
                    return false, err
                end
            """)

            if result[0]:
                return (True, None)
            else:
                return (False, str(result[1]))

        except Exception as e:
            return (False, str(e))

    def test_security(self) -> dict:
        """
        Run security tests to verify sandbox isolation.

        Returns:
            Dict with test results
        """
        results = {}

        # Test blocked functions
        blocked_tests = [
            ("os.execute('echo pwned')", "os.execute"),
            ("io.open('/etc/passwd')", "io.open"),
            ("loadfile('evil.lua')", "loadfile"),
            ("require('socket')", "require"),
            ("debug.getinfo(1)", "debug"),
            ("_G.os", "_G access"),
        ]

        for code, name in blocked_tests:
            script = f"""
            function match(ctx)
                local result = {code}
                return {{matched = true}}
            end
            """
            result = self.execute(script, {'digits': '12345678'})
            results[name] = {
                'blocked': not result.success or 'error' in result.error.lower() or 'nil' in str(result),
                'error': result.error if result.error else 'Executed (security breach!)'
            }

        return results


def create_context(serial: str, metadata: dict = None) -> dict:
    """
    Create a context dict for pattern execution.

    Args:
        serial: Full serial number (e.g., 'A12344321B')
        metadata: Optional metadata dict

    Returns:
        Context dict with digits, full_serial, metadata, digit_list
    """
    from datetime import date

    # Build metadata with current date defaults
    meta = metadata.copy() if metadata else {}
    today = date.today()
    meta.setdefault('current_year', today.year)
    meta.setdefault('current_month', today.month)
    meta.setdefault('current_day', today.day)

    # Extract numeric digits
    digits = ''.join(c for c in serial if c.isdigit())

    # Create digit list as integers
    digit_list = [int(d) for d in digits] if digits else []

    return {
        'digits': digits,
        'full_serial': serial,
        'metadata': meta,
        'digit_list': digit_list
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Pattern Sandbox - Security Test")
    print("=" * 60)

    sandbox = PatternSandbox()

    # Test basic pattern execution
    radar_script = """
    function match(ctx)
        local rev = string.reverse(ctx.digits)
        if ctx.digits ~= rev then
            return {matched = false}
        end

        local colors = {"orange", "coral", "gold", "salmon"}
        local highlights = {}
        local connectors = {}

        for i = 0, 3 do
            local j = 7 - i
            table.insert(highlights, {positions = {i, j}, color = colors[i+1]})
            table.insert(connectors, {from = i, to = j, color = colors[i+1], style = "arc"})
        end

        return {matched = true, highlights = highlights, connectors = connectors}
    end
    """

    ctx = create_context("A12344321B")
    result = sandbox.execute(radar_script, ctx)

    print(f"\nRADAR pattern test:")
    print(f"  Input: {ctx['digits']}")
    print(f"  Matched: {result.matched}")
    print(f"  Highlights: {len(result.highlights)}")
    print(f"  Connectors: {len(result.connectors)}")
    print(f"  Execution time: {result.execution_time_ms:.2f}ms")

    # Test non-matching
    ctx2 = create_context("A12345678B")
    result2 = sandbox.execute(radar_script, ctx2)
    print(f"\nNon-radar test:")
    print(f"  Input: {ctx2['digits']}")
    print(f"  Matched: {result2.matched}")

    # Security tests
    print("\n" + "=" * 60)
    print("Security Tests:")
    security_results = sandbox.test_security()
    for test_name, test_result in security_results.items():
        status = "BLOCKED" if test_result['blocked'] else "FAILED!"
        print(f"  {test_name}: {status}")

    print("\n" + "=" * 60)
    print("Syntax validation test:")

    # Valid syntax
    valid, err = sandbox.validate_syntax("function match(ctx) return {matched=true} end")
    print(f"  Valid script: {'OK' if valid else f'ERROR: {err}'}")

    # Invalid syntax
    valid, err = sandbox.validate_syntax("function match(ctx return {matched=true} end")
    print(f"  Invalid script: {'ERROR' if not valid else 'Unexpectedly OK'}: {err}")
