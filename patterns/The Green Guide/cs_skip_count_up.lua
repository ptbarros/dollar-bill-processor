--[[
Pattern: CS_SKIP_COUNT_UP
DisplayName: CS-Skip Count Up Note
Description: A CS-Skip Note where the non-skip positions count up in sequence. e.g., M x1x2x3x4 M or M 1x2x3x4x M.
BookRef: CS-1610
Tier: 4
Examples: ["91929394", "01020304", "10203040"]
Odds: 1 in 140
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Check if 4 positions form ascending consecutive integers: n, n+1, n+2, n+3
    local function is_ascending_seq(p1, p2, p3, p4)
        local n1 = tonumber(d:sub(p1, p1))
        local n2 = tonumber(d:sub(p2, p2))
        local n3 = tonumber(d:sub(p3, p3))
        local n4 = tonumber(d:sub(p4, p4))
        return n2 == n1 + 1 and n3 == n2 + 1 and n4 == n3 + 1
    end

    -- Check if 4 positions are all the same digit (the skip/repeat digit)
    local function is_all_same(p1, p2, p3, p4)
        local s = d:sub(p1, p1)
        return d:sub(p2, p2) == s and d:sub(p3, p3) == s and d:sub(p4, p4) == s
    end

    -- Variant 1: odd positions (1,3,5,7) count up, even positions (2,4,6,8) are same
    if is_ascending_seq(1, 3, 5, 7) and is_all_same(2, 4, 6, 8) then
        local skip_d = d:sub(2, 2)
        local count_start = d:sub(1, 1)
        return {
            matched = true,
            highlights = {
                {positions = {0, 2, 4, 6}, color = "lime"},
                {positions = {1, 3, 5, 7}, color = "gray"}
            },
            message = count_start .. ".." .. d:sub(7,7) .. " ascending at odd positions (CS-Skip Count Up)"
        }
    end

    -- Variant 2: even positions (2,4,6,8) count up, odd positions (1,3,5,7) are same
    if is_ascending_seq(2, 4, 6, 8) and is_all_same(1, 3, 5, 7) then
        local skip_d = d:sub(1, 1)
        local count_start = d:sub(2, 2)
        return {
            matched = true,
            highlights = {
                {positions = {1, 3, 5, 7}, color = "lime"},
                {positions = {0, 2, 4, 6}, color = "gray"}
            },
            message = count_start .. ".." .. d:sub(8,8) .. " ascending at even positions (CS-Skip Count Up)"
        }
    end

    return {matched = false}
end
