--[[
Pattern: THREE_PAIRS_LADDER
Description: 3 pairs forming ladder (AABBCCXX)
Tier: 4
Examples: ["11223345", "22334456"]
Odds: 1 in 27,777
Price: $10-$50
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check for AA BB CC pattern where A, B, C form a ladder
    -- First pair
    local a = digits:sub(1, 1)
    if digits:sub(2, 2) ~= a then
        return {matched = false}
    end

    -- Second pair
    local b = digits:sub(3, 3)
    if digits:sub(4, 4) ~= b then
        return {matched = false}
    end

    -- Third pair
    local c = digits:sub(5, 5)
    if digits:sub(6, 6) ~= c then
        return {matched = false}
    end

    -- Check A, B, C form a ladder
    local a_num = tonumber(a)
    local b_num = tonumber(b)
    local c_num = tonumber(c)

    local ascending = (b_num == a_num + 1) and (c_num == b_num + 1)
    local descending = (b_num == a_num - 1) and (c_num == b_num - 1)

    if not ascending and not descending then
        return {matched = false}
    end

    local direction = ascending and "ascending" or "descending"

    return {
        matched = true,
        highlights = {
            highlight({0, 1}, "lime", "pair A"),
            highlight({2, 3}, "teal", "pair B"),
            highlight({4, 5}, "cyan", "pair C")
        },
        group_boxes = {
            {from = 0, to = 1, color = "lime", thickness = 2},
            {from = 2, to = 3, color = "teal", thickness = 2},
            {from = 4, to = 5, color = "cyan", thickness = 2}
        },
        connectors = {
            connector(0, 2, "lime", "line"),
            connector(2, 4, "lime", "line")
        },
        message = "Three pairs ladder " .. direction
    }
end
