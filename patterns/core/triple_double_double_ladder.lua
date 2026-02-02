--[[
Pattern: TRIPLE_DOUBLE_DOUBLE_LADDER
Description: Triple + double + double ladder (AAABBCCX)
Tier: 6
Examples: ["11122334", "22233445"]
Odds: 1 in 90,071
Price: $40-$200+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check for AAA BB CC X pattern where A, B, C form a ladder
    -- First triple
    local a = digits:sub(1, 1)
    if digits:sub(2, 2) ~= a or digits:sub(3, 3) ~= a then
        return {matched = false}
    end

    -- Second double
    local b = digits:sub(4, 4)
    if digits:sub(5, 5) ~= b then
        return {matched = false}
    end

    -- Third double
    local c = digits:sub(6, 6)
    if digits:sub(7, 7) ~= c then
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
            highlight({0, 1, 2}, "lime", "triple"),
            highlight({3, 4}, "teal", "double B"),
            highlight({5, 6}, "cyan", "double C")
        },
        group_boxes = {
            {from = 0, to = 2, color = "lime", thickness = 2},
            {from = 3, to = 4, color = "teal", thickness = 2},
            {from = 5, to = 6, color = "cyan", thickness = 2}
        },
        connectors = {},
        message = "Triple + double + double ladder " .. direction
    }
end
