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
