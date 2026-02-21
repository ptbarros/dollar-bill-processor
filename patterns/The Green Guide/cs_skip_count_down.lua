--[[
Pattern: CS_SKIP_COUNT_DOWN
DisplayName: CS-Skip Count Down Note
Description: A CS-Skip Note where the non-skip positions count down in sequence. e.g., M x4x3x2x1 M or M 4x3x2x1x M.
BookRef: CS-1620
Tier: 4
Examples: ["94939291", "40302010", "04030201"]
Odds: 1 in 140
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Check if 4 positions form descending consecutive integers: n, n-1, n-2, n-3
    local function is_descending_seq(p1, p2, p3, p4)
        local n1 = tonumber(d:sub(p1, p1))
        local n2 = tonumber(d:sub(p2, p2))
        local n3 = tonumber(d:sub(p3, p3))
        local n4 = tonumber(d:sub(p4, p4))
        return n2 == n1 - 1 and n3 == n2 - 1 and n4 == n3 - 1
    end

    local function is_all_same(p1, p2, p3, p4)
        local s = d:sub(p1, p1)
        return d:sub(p2, p2) == s and d:sub(p3, p3) == s and d:sub(p4, p4) == s
    end

    -- Variant 1: odd positions (1,3,5,7) count down, even positions are same
    if is_descending_seq(1, 3, 5, 7) and is_all_same(2, 4, 6, 8) then
        local count_start = d:sub(1, 1)
        return {
            matched = true,
            highlights = {
                {positions = {0, 2, 4, 6}, color = "lime"},
                {positions = {1, 3, 5, 7}, color = "gray"}
            },
            message = count_start .. ".." .. d:sub(7,7) .. " descending at odd positions (CS-Skip Count Down)"
        }
    end

    -- Variant 2: even positions (2,4,6,8) count down, odd positions are same
    if is_descending_seq(2, 4, 6, 8) and is_all_same(1, 3, 5, 7) then
        local count_start = d:sub(2, 2)
        return {
            matched = true,
            highlights = {
                {positions = {1, 3, 5, 7}, color = "lime"},
                {positions = {0, 2, 4, 6}, color = "gray"}
            },
            message = count_start .. ".." .. d:sub(8,8) .. " descending at even positions (CS-Skip Count Down)"
        }
    end

    return {matched = false}
end
