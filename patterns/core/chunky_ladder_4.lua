--[[
Pattern: CHUNKY_LADDER_4
Description: 4-digit chunky ladder (AAABCCDD)
Tier: 6
Examples: ["11123344", "22234455"]
Odds: 1 in 217,194
Price: $50-$700+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Pattern: AAABCCDD where A, B, C, D form ladder
    -- Check first 3 are same (A)
    local a = digits:sub(1, 1)
    if digits:sub(2, 2) ~= a or digits:sub(3, 3) ~= a then
        return {matched = false}
    end

    -- Get B, C, D values
    local b = digits:sub(4, 4)
    local c = digits:sub(5, 5)
    local d = digits:sub(7, 7)

    -- C should be double (CC)
    if digits:sub(6, 6) ~= c then
        return {matched = false}
    end

    -- D should be double (DD)
    if digits:sub(8, 8) ~= d then
        return {matched = false}
    end

    -- Check A, B, C, D form a ladder
    local nums = {tonumber(a), tonumber(b), tonumber(c), tonumber(d)}
    local ascending = true
    local descending = true
    for i = 1, 3 do
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
            highlight({3}, "teal", "B"),
            highlight({4, 5}, "cyan", "C pair"),
            highlight({6, 7}, "blue", "D pair")
        },
        group_boxes = {
            {from = 0, to = 2, color = "lime", thickness = 2},
            {from = 4, to = 5, color = "cyan", thickness = 2},
            {from = 6, to = 7, color = "blue", thickness = 2}
        },
        connectors = {},
        message = "Chunky ladder 4 " .. direction
    }
end
