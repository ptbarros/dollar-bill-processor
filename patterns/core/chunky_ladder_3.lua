--[[
Pattern: CHUNKY_LADDER_3
Description: 3-digit chunky ladder (AAAABCCC)
Tier: 6
Examples: ["11112333", "22223444"]
Odds: 1 in 426,666
Price: $40-$350
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Pattern: AAAABCCC where A, B, C are consecutive
    -- Check first 4 are same
    local a = digits:sub(1, 1)
    if digits:sub(2, 2) ~= a or digits:sub(3, 3) ~= a or digits:sub(4, 4) ~= a then
        return {matched = false}
    end

    -- Check position 5 (B)
    local b = digits:sub(5, 5)
    local a_num = tonumber(a)
    local b_num = tonumber(b)

    -- B should be A+1 or A-1
    local ascending = (b_num == a_num + 1)
    local descending = (b_num == a_num - 1)
    if not ascending and not descending then
        return {matched = false}
    end

    -- Check last 3 are same (C)
    local c = digits:sub(6, 6)
    if digits:sub(7, 7) ~= c or digits:sub(8, 8) ~= c then
        return {matched = false}
    end

    -- C should continue the sequence
    local c_num = tonumber(c)
    if ascending and c_num ~= b_num + 1 then
        return {matched = false}
    end
    if descending and c_num ~= b_num - 1 then
        return {matched = false}
    end

    local direction = ascending and "ascending" or "descending"

    return {
        matched = true,
        highlights = {
            highlight({0, 1, 2, 3}, "lime", "A group"),
            highlight({4}, "teal", "B"),
            highlight({5, 6, 7}, "cyan", "C group")
        },
        group_boxes = {
            {from = 0, to = 3, color = "lime", thickness = 2},
            {from = 5, to = 7, color = "cyan", thickness = 2}
        },
        connectors = {},
        message = "Chunky ladder 3 " .. direction
    }
end
