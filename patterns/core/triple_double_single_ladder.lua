--[[
Pattern: TRIPLE_DOUBLE_SINGLE_LADDER
Description: Triple + double + single ladder consecutive
Tier: 4
Examples: ["11122345", "22233456"]
Odds: 1 in 27,777
Price: $5-$40
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check for AAA BB C D E pattern where A, B, C, D, E form a ladder
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

    -- Singles
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
            highlight({0, 1, 2}, "lime", "triple"),
            highlight({3, 4}, "teal", "double"),
            highlight({5}, "cyan", "single"),
            highlight({6}, "blue", "single"),
            highlight({7}, "purple", "single")
        },
        group_boxes = {
            {from = 0, to = 2, color = "lime", thickness = 2},
            {from = 3, to = 4, color = "teal", thickness = 2}
        },
        connectors = {},
        message = "Triple + double + singles ladder " .. direction
    }
end
