--[[
    Pattern Helper Library

    Common utility functions available to all pattern scripts.
    These are automatically loaded into the sandbox environment.

    Usage in patterns:
        local counts = count_digits(ctx.digits)
        local runs = find_runs(ctx.digits)
--]]

-- Count occurrences of each digit
-- Returns table like {["0"]=2, ["1"]=3, ...}
function count_digits(s)
    local counts = {}
    for i = 1, #s do
        local d = s:sub(i, i)
        counts[d] = (counts[d] or 0) + 1
    end
    return counts
end

-- Find consecutive runs of same digit
-- Returns list of {digit, start, length} (0-indexed start)
function find_runs(s)
    local runs = {}
    local i = 1
    while i <= #s do
        local d = s:sub(i, i)
        local run_start = i - 1  -- Convert to 0-indexed
        local run_len = 1
        while i + run_len <= #s and s:sub(i + run_len, i + run_len) == d do
            run_len = run_len + 1
        end
        table.insert(runs, {digit = d, start = run_start, length = run_len})
        i = i + run_len
    end
    return runs
end

-- Check if string contains only specified digits
-- allowed is a string like "01" or "01689"
function only_digits(s, allowed)
    local allowed_set = {}
    for i = 1, #allowed do
        allowed_set[allowed:sub(i, i)] = true
    end
    for i = 1, #s do
        if not allowed_set[s:sub(i, i)] then
            return false
        end
    end
    return true
end

-- Check if digits form ascending or descending ladder
function is_ladder(s)
    if #s < 2 then return false end
    local ascending = true
    local descending = true
    for i = 1, #s - 1 do
        local curr = tonumber(s:sub(i, i))
        local next = tonumber(s:sub(i + 1, i + 1))
        if next ~= curr + 1 then ascending = false end
        if next ~= curr - 1 then descending = false end
    end
    return ascending or descending
end

-- Check if ascending ladder
function is_ascending(s)
    if #s < 2 then return false end
    for i = 1, #s - 1 do
        local curr = tonumber(s:sub(i, i))
        local next = tonumber(s:sub(i + 1, i + 1))
        if next ~= curr + 1 then return false end
    end
    return true
end

-- Check if descending ladder
function is_descending(s)
    if #s < 2 then return false end
    for i = 1, #s - 1 do
        local curr = tonumber(s:sub(i, i))
        local next = tonumber(s:sub(i + 1, i + 1))
        if next ~= curr - 1 then return false end
    end
    return true
end

-- Build highlight entry for given positions with color
-- positions: list of 0-indexed positions
-- color: color name string
-- label: optional label string
function highlight(positions, color, label)
    local h = {positions = positions, color = color}
    if label then h.label = label end
    return h
end

-- Build connector entry between two positions
-- from, to: 0-indexed positions
-- color: color name string
-- style: optional style ("arc", "line", "dashed")
function connector(from_pos, to_pos, color, style)
    local c = {from = from_pos, to = to_pos, color = color}
    if style then c.style = style end
    return c
end

-- Highlight a range of consecutive positions
-- start, stop: 0-indexed positions (inclusive)
-- color: color name string
function highlight_range(start_pos, stop_pos, color, label)
    local positions = {}
    for i = start_pos, stop_pos do
        table.insert(positions, i)
    end
    return highlight(positions, color, label)
end

-- Get most common digit(s)
-- Returns digit, count
function most_common(s)
    local counts = count_digits(s)
    local max_digit = nil
    local max_count = 0
    for d, c in pairs(counts) do
        if c > max_count then
            max_digit = d
            max_count = c
        end
    end
    return max_digit, max_count
end

-- Count unique digits in string
function unique_count(s)
    local seen = {}
    local count = 0
    for i = 1, #s do
        local d = s:sub(i, i)
        if not seen[d] then
            seen[d] = true
            count = count + 1
        end
    end
    return count
end

-- Sum of all digits
function digit_sum(s)
    local sum = 0
    for i = 1, #s do
        sum = sum + tonumber(s:sub(i, i))
    end
    return sum
end

-- Check if string is palindrome
function is_palindrome(s)
    return s == string.reverse(s)
end

-- Flipper digit mappings
FLIP_MAP = {
    ["0"] = "0", ["1"] = "1", ["6"] = "9", ["8"] = "8", ["9"] = "6"
}
FLIP_VALID = {["0"] = true, ["1"] = true, ["6"] = true, ["8"] = true, ["9"] = true}

-- Check if all digits are flip-valid
function all_flip_valid(s)
    for i = 1, #s do
        if not FLIP_VALID[s:sub(i, i)] then
            return false
        end
    end
    return true
end

-- Get flipped version of string (rotated 180 degrees)
function flip_string(s)
    if not all_flip_valid(s) then return nil end
    local result = {}
    for i = #s, 1, -1 do
        table.insert(result, FLIP_MAP[s:sub(i, i)])
    end
    return table.concat(result)
end

-- Find consecutive pairs (AA, BB, etc.)
-- Returns list of {digit, start} (0-indexed start)
function find_pairs(s)
    local pairs = {}
    local i = 1
    while i < #s do
        if s:sub(i, i) == s:sub(i + 1, i + 1) then
            table.insert(pairs, {digit = s:sub(i, i), start = i - 1})
            i = i + 2  -- Skip past the pair
        else
            i = i + 1
        end
    end
    return pairs
end

-- Color palette constants for consistency
COLORS = {
    -- Primary colors for pattern types
    purple = "purple",    -- Flipper-valid digits
    blue = "blue",        -- Binary patterns
    cyan = "cyan",        -- Trinary patterns
    orange = "orange",    -- Primary pairs (radar)
    coral = "coral",      -- Secondary pairs
    gold = "gold",        -- Quads/runs
    salmon = "salmon",    -- Tertiary pairs
    magenta = "magenta",  -- Repeater
    yellow = "yellow",    -- Solid/dominant
    lime = "lime",        -- Ladder sequence
    teal = "teal",        -- Double pairs
    red = "red",          -- Errors/broken patterns
}

-- =============================================================================
-- LADDER HELPERS
-- =============================================================================

-- Find ladder (ascending or descending) of given length anywhere in string
-- Returns: {found=bool, start=pos, length=len, ascending=bool} or nil
function find_ladder_of_length(s, min_length)
    local n = #s
    if n < min_length then return nil end

    for start = 1, n - min_length + 1 do
        -- Try ascending
        local asc_len = 1
        for i = start, n - 1 do
            local curr = tonumber(s:sub(i, i))
            local next = tonumber(s:sub(i + 1, i + 1))
            if next == curr + 1 then
                asc_len = asc_len + 1
            else
                break
            end
        end
        if asc_len >= min_length then
            return {found = true, start = start - 1, length = asc_len, ascending = true}
        end

        -- Try descending
        local desc_len = 1
        for i = start, n - 1 do
            local curr = tonumber(s:sub(i, i))
            local next = tonumber(s:sub(i + 1, i + 1))
            if next == curr - 1 then
                desc_len = desc_len + 1
            else
                break
            end
        end
        if desc_len >= min_length then
            return {found = true, start = start - 1, length = desc_len, ascending = false}
        end
    end
    return nil
end

-- Find the longest ladder in the string
function find_longest_ladder(s)
    local best = nil
    local n = #s

    local i = 1
    while i <= n do
        -- Try ascending from position i
        local asc_len = 1
        local j = i
        while j < n do
            local curr = tonumber(s:sub(j, j))
            local next = tonumber(s:sub(j + 1, j + 1))
            if next == curr + 1 then
                asc_len = asc_len + 1
                j = j + 1
            else
                break
            end
        end
        if asc_len >= 2 and (not best or asc_len > best.length) then
            best = {start = i - 1, length = asc_len, ascending = true}
        end

        -- Try descending from position i
        local desc_len = 1
        j = i
        while j < n do
            local curr = tonumber(s:sub(j, j))
            local next = tonumber(s:sub(j + 1, j + 1))
            if next == curr - 1 then
                desc_len = desc_len + 1
                j = j + 1
            else
                break
            end
        end
        if desc_len >= 2 and (not best or desc_len > best.length) then
            best = {start = i - 1, length = desc_len, ascending = false}
        end

        i = i + 1
    end
    return best
end

-- =============================================================================
-- RUN/CONSECUTIVE HELPERS
-- =============================================================================

-- Check for N consecutive identical digits anywhere in string
-- Returns: {found=bool, digit=char, start=pos, length=len} or nil
function has_n_consecutive(s, n)
    local runs = find_runs(s)
    for _, run in ipairs(runs) do
        if run.length >= n then
            return {found = true, digit = run.digit, start = run.start, length = run.length}
        end
    end
    return nil
end

-- Find longest run of consecutive identical digits
function find_longest_run(s)
    local runs = find_runs(s)
    local best = nil
    for _, run in ipairs(runs) do
        if not best or run.length > best.length then
            best = run
        end
    end
    return best
end

-- =============================================================================
-- COUNTING PATTERN HELPERS
-- =============================================================================

-- Check if string matches a counting pattern with two-digit pairs
-- step: increment between pairs (e.g., 10 for 10,20,30,40)
-- Returns: {matched=bool, start_value=num} or nil
function is_counting_pairs(s, step)
    if #s ~= 8 then return nil end

    -- Parse as four 2-digit numbers
    local nums = {}
    for i = 1, 4 do
        local pair = s:sub((i-1)*2 + 1, i*2)
        local num = tonumber(pair)
        if not num then return nil end
        table.insert(nums, num)
    end

    -- Check if they form arithmetic sequence with given step
    for i = 1, 3 do
        if nums[i+1] - nums[i] ~= step then
            return nil
        end
    end

    return {matched = true, start_value = nums[1]}
end

-- Check for counting ladder pattern (like 10203040)
-- Returns: matched, message
function check_counting_ladder(s)
    if #s ~= 8 then return false, nil end

    -- Check X0Y0Z0W0 pattern where X,Y,Z,W are consecutive
    if s:sub(2,2) == "0" and s:sub(4,4) == "0" and s:sub(6,6) == "0" and s:sub(8,8) == "0" then
        local d1 = tonumber(s:sub(1,1))
        local d2 = tonumber(s:sub(3,3))
        local d3 = tonumber(s:sub(5,5))
        local d4 = tonumber(s:sub(7,7))
        if d2 == d1 + 1 and d3 == d2 + 1 and d4 == d3 + 1 then
            return true, string.format("%d0%d0%d0%d0", d1, d2, d3, d4)
        end
    end
    return false, nil
end

-- =============================================================================
-- DIGIT ANALYSIS HELPERS
-- =============================================================================

-- Get sorted unique digits as string
function get_unique_digits(s)
    local seen = {}
    local digits = {}
    for i = 1, #s do
        local d = s:sub(i, i)
        if not seen[d] then
            seen[d] = true
            table.insert(digits, d)
        end
    end
    table.sort(digits)
    return table.concat(digits)
end

-- Check if broken palindrome (exactly N mismatches)
function is_broken_palindrome(s, max_mismatches)
    local n = #s
    local mismatches = 0
    local mismatch_positions = {}

    for i = 1, n // 2 do
        if s:sub(i, i) ~= s:sub(n - i + 1, n - i + 1) then
            mismatches = mismatches + 1
            table.insert(mismatch_positions, {i - 1, n - i})  -- 0-indexed
        end
    end

    if mismatches >= 1 and mismatches <= max_mismatches then
        return {matched = true, mismatches = mismatches, positions = mismatch_positions}
    end
    return nil
end

-- Check if string starts with given prefix
function starts_with(s, prefix)
    return s:sub(1, #prefix) == prefix
end

-- Check if string ends with given suffix
function ends_with(s, suffix)
    return s:sub(-#suffix) == suffix
end

-- Check if string contains substring
function contains(s, substr)
    return string.find(s, substr, 1, true) ~= nil
end

-- =============================================================================
-- REPEATER HELPERS
-- =============================================================================

-- Check if string is a repeater (ABCDABCD)
function is_repeater(s)
    if #s ~= 8 then return false end
    return s:sub(1, 4) == s:sub(5, 8)
end

-- Check if string is a super repeater (ABABABAB)
function is_super_repeater(s)
    if #s ~= 8 then return false end
    local pair = s:sub(1, 2)
    return s == pair .. pair .. pair .. pair
end

-- =============================================================================
-- POSITION FINDING HELPERS
-- =============================================================================

-- Get all positions of a digit in string (0-indexed)
function find_digit_positions(s, digit)
    local positions = {}
    for i = 1, #s do
        if s:sub(i, i) == digit then
            table.insert(positions, i - 1)
        end
    end
    return positions
end

-- Get positions of all digits matching a set
function find_matching_positions(s, digit_set)
    local positions = {}
    for i = 1, #s do
        if digit_set[s:sub(i, i)] then
            table.insert(positions, i - 1)
        end
    end
    return positions
end

-- =============================================================================
-- PAIR/GROUP HELPERS
-- =============================================================================

-- Check for consecutive pairs pattern (AABBCCDD or partial)
-- Returns: list of pairs with their positions
function find_consecutive_pairs(s)
    local pairs = {}
    local i = 1
    while i < #s do
        if s:sub(i, i) == s:sub(i + 1, i + 1) then
            table.insert(pairs, {
                digit = s:sub(i, i),
                start = i - 1,  -- 0-indexed
                length = 2
            })
            i = i + 2
        else
            i = i + 1
        end
    end
    return pairs
end

-- Check for four consecutive pairs (AABBCCDD)
function has_four_consecutive_pairs(s)
    if #s ~= 8 then return false end
    for i = 1, 4 do
        local pos = (i - 1) * 2 + 1
        if s:sub(pos, pos) ~= s:sub(pos + 1, pos + 1) then
            return false
        end
    end
    return true
end

-- Check for three consecutive pairs at start (AABBCC??)
function has_three_consecutive_pairs_start(s)
    if #s < 6 then return false end
    for i = 1, 3 do
        local pos = (i - 1) * 2 + 1
        if s:sub(pos, pos) ~= s:sub(pos + 1, pos + 1) then
            return false
        end
    end
    return true
end

-- Count total pairs in string (not necessarily consecutive)
function count_pairs(s)
    local counts = count_digits(s)
    local total_pairs = 0
    for _, count in pairs(counts) do
        total_pairs = total_pairs + (count // 2)
    end
    return total_pairs
end

-- =============================================================================
-- BOOKEND HELPERS
-- =============================================================================

-- Check if first N and last N digits match
function is_bookended(s, n)
    if #s < 2 * n then return false end
    return s:sub(1, n) == s:sub(-n)
end

-- =============================================================================
-- STEP/ALTERNATING HELPERS
-- =============================================================================

-- Check if string alternates between two values (XYXYXYXY)
function is_alternating(s)
    if #s < 2 then return false end
    local a = s:sub(1, 1)
    local b = s:sub(2, 2)
    if a == b then return false end

    for i = 1, #s do
        local expected = (i % 2 == 1) and a or b
        if s:sub(i, i) ~= expected then
            return false
        end
    end
    return true
end

-- Check step ladder (constant step between digits)
function check_step_ladder(s, step)
    if #s < 2 then return false end

    local first = tonumber(s:sub(1, 1))
    for i = 2, #s do
        local curr = tonumber(s:sub(i, i))
        local expected = (first + (i - 1) * step) % 10
        if curr ~= expected then
            return false
        end
    end
    return true
end

-- =============================================================================
-- TRIPLES HELPERS
-- =============================================================================

-- Find triples (3 consecutive identical) in string
function find_triples(s)
    local triples = {}
    local i = 1
    while i <= #s - 2 do
        local d = s:sub(i, i)
        if s:sub(i+1, i+1) == d and s:sub(i+2, i+2) == d then
            local len = 3
            while i + len <= #s and s:sub(i + len, i + len) == d do
                len = len + 1
            end
            table.insert(triples, {digit = d, start = i - 1, length = len})
            i = i + len
        else
            i = i + 1
        end
    end
    return triples
end

-- Find quads (4+ consecutive identical) in string
function find_quads(s)
    local quads = {}
    local runs = find_runs(s)
    for _, run in ipairs(runs) do
        if run.length >= 4 then
            table.insert(quads, run)
        end
    end
    return quads
end

-- =============================================================================
-- DATE/CALENDAR HELPERS
-- =============================================================================

-- Check if a year is a leap year
function is_leap_year(y)
    return (y % 4 == 0 and y % 100 ~= 0) or (y % 400 == 0)
end

-- Get number of days in a given month/year
function days_in_month(m, y)
    local dim = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
    if m == 2 and is_leap_year(y) then return 29 end
    return dim[m] or 0
end

-- Check if mm/dd/yyyy is a valid calendar date (year 1700-2099)
function is_valid_date(mm, dd, yyyy)
    if yyyy < 1700 or yyyy > 2099 then return false end
    if mm < 1 or mm > 12 then return false end
    if dd < 1 or dd > days_in_month(mm, yyyy) then return false end
    return true
end

-- Check if year is in valid range (1700-2099 for year note patterns)
function is_valid_year(y)
    return y >= 1700 and y <= 2099
end

-- Check if mm/dd is a plausible calendar day (ignoring year; Feb allows 29)
function is_valid_mmdd(mm, dd)
    if mm < 1 or mm > 12 then return false end
    if dd < 1 or dd > 31 then return false end
    local max_days = {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
    return dd <= max_days[mm]
end
