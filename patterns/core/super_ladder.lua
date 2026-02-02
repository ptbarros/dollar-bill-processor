--[[
Pattern: SUPER_LADDER
Description: Super ladder patterns (AAABBBCC, etc.)
Tier: 7
Examples: ["11122233", "22233344"]
Odds: 1 in 941,176
Price: $40-$300
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check for AAABBBCC pattern where A, B, C are consecutive
    -- First 3 same (AAA)
    local a = digits:sub(1, 1)
    if digits:sub(2, 2) ~= a or digits:sub(3, 3) ~= a then
        return {matched = false}
    end

    -- Next 3 same (BBB)
    local b = digits:sub(4, 4)
    if digits:sub(5, 5) ~= b or digits:sub(6, 6) ~= b then
        return {matched = false}
    end

    -- Last 2 same (CC)
    local c = digits:sub(7, 7)
    if digits:sub(8, 8) ~= c then
        return {matched = false}
    end

    -- Check A, B, C are consecutive
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
            highlight({0, 1, 2}, "lime", "A group"),
            highlight({3, 4, 5}, "teal", "B group"),
            highlight({6, 7}, "cyan", "C pair")
        },
        group_boxes = {
            {from = 0, to = 2, color = "lime", thickness = 2},
            {from = 3, to = 5, color = "teal", thickness = 2},
            {from = 6, to = 7, color = "cyan", thickness = 2}
        },
        connectors = {},
        message = "Super ladder " .. direction .. ": " .. a .. a .. a .. b .. b .. b .. c .. c
    }
end
