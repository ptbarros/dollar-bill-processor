--[[
Pattern: CHUNKY_LADDER_7
Description: 7-digit chunky ladder (ABCDEFGG)
Tier: 6
Examples: ["12345677", "23456788"]
Odds: 1 in 1,959,183
Price: $100-$1,500+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Pattern: ABCDEFGG where A, B, C, D, E, F, G form ladder
    -- Check GG at end
    local g = digits:sub(7, 7)
    if digits:sub(8, 8) ~= g then
        return {matched = false}
    end

    -- Get A through G
    local nums = {}
    for i = 1, 7 do
        table.insert(nums, tonumber(digits:sub(i, i)))
    end

    -- Check A, B, C, D, E, F, G form a ladder
    local ascending = true
    local descending = true
    for i = 1, 6 do
        if nums[i + 1] ~= nums[i] + 1 then ascending = false end
        if nums[i + 1] ~= nums[i] - 1 then descending = false end
    end

    if not ascending and not descending then
        return {matched = false}
    end

    local direction = ascending and "ascending" or "descending"

    return {
        matched = true,
        highlights = {
            highlight({0, 1, 2, 3, 4, 5}, "lime", "ladder"),
            highlight({6, 7}, "gold", "double")
        },
        group_boxes = {
            {from = 6, to = 7, color = "gold", thickness = 2}
        },
        connectors = {},
        message = "Chunky ladder 7 " .. direction
    }
end
