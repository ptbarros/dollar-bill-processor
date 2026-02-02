--[[
Pattern: CHUNKY_LADDER_6
Description: 6-digit chunky ladder (AABBCDEF)
Tier: 6
Examples: ["11223456", "22334567"]
Odds: 1 in 507,936
Price: $40-$350
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Pattern: AABBCDEF where A, B, C, D, E, F form ladder
    -- Check AA
    local a = digits:sub(1, 1)
    if digits:sub(2, 2) ~= a then
        return {matched = false}
    end

    -- Check BB
    local b = digits:sub(3, 3)
    if digits:sub(4, 4) ~= b then
        return {matched = false}
    end

    -- Get C, D, E, F
    local c = digits:sub(5, 5)
    local d = digits:sub(6, 6)
    local e = digits:sub(7, 7)
    local f = digits:sub(8, 8)

    -- Check A, B, C, D, E, F form a ladder
    local nums = {tonumber(a), tonumber(b), tonumber(c), tonumber(d), tonumber(e), tonumber(f)}
    local ascending = true
    local descending = true
    for i = 1, 5 do
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
            highlight({0, 1}, "lime", "A pair"),
            highlight({2, 3}, "teal", "B pair"),
            highlight({4}, "cyan", "C"),
            highlight({5}, "blue", "D"),
            highlight({6}, "purple", "E"),
            highlight({7}, "magenta", "F")
        },
        group_boxes = {
            {from = 0, to = 1, color = "lime", thickness = 2},
            {from = 2, to = 3, color = "teal", thickness = 2}
        },
        connectors = {},
        message = "Chunky ladder 6 " .. direction
    }
end
