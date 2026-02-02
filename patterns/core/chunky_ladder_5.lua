--[[
Pattern: CHUNKY_LADDER_5
Description: 5-digit chunky ladder (AAABBCDE)
Tier: 6
Examples: ["11122345", "22233456"]
Odds: 1 in 396,694
Price: $40-$350
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Pattern: AAABBCDE where A, B, C, D, E form ladder
    -- Check first 3 are same (A)
    local a = digits:sub(1, 1)
    if digits:sub(2, 2) ~= a or digits:sub(3, 3) ~= a then
        return {matched = false}
    end

    -- Check BB
    local b = digits:sub(4, 4)
    if digits:sub(5, 5) ~= b then
        return {matched = false}
    end

    -- Get C, D, E
    local c = digits:sub(6, 6)
    local d = digits:sub(7, 7)
    local e = digits:sub(8, 8)

    -- Check A, B, C, D, E form a ladder
    local nums = {tonumber(a), tonumber(b), tonumber(c), tonumber(d), tonumber(e)}
    local ascending = true
    local descending = true
    for i = 1, 4 do
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
            highlight({0, 1, 2}, "lime", "A group"),
            highlight({3, 4}, "teal", "B pair"),
            highlight({5}, "cyan", "C"),
            highlight({6}, "blue", "D"),
            highlight({7}, "purple", "E")
        },
        group_boxes = {
            {from = 0, to = 2, color = "lime", thickness = 2},
            {from = 3, to = 4, color = "teal", thickness = 2}
        },
        connectors = {},
        message = "Chunky ladder 5 " .. direction
    }
end
