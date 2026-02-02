--[[
Pattern: DOUBLES_LADDER
Description: Pairs in ascending/descending order (AABBCCDD)
Tier: 1
Examples: ["11223344", "22334455", "99887766"]
Odds: 1 in 6,400,000
Price: $100-$4,500
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check for AABBCCDD pattern where A,B,C,D are consecutive
    -- First verify pairs
    for i = 1, 4 do
        local pos = (i - 1) * 2 + 1
        if digits:sub(pos, pos) ~= digits:sub(pos + 1, pos + 1) then
            return {matched = false}
        end
    end

    -- Get the four values
    local vals = {}
    for i = 1, 4 do
        local pos = (i - 1) * 2 + 1
        table.insert(vals, tonumber(digits:sub(pos, pos)))
    end

    -- Check if ascending or descending
    local ascending = true
    local descending = true
    for i = 1, 3 do
        if vals[i + 1] ~= vals[i] + 1 then ascending = false end
        if vals[i + 1] ~= vals[i] - 1 then descending = false end
    end

    if not ascending and not descending then
        return {matched = false}
    end

    local direction = ascending and "ascending" or "descending"

    return {
        matched = true,
        highlights = {
            highlight({0, 1}, "lime", "pair A"),
            highlight({2, 3}, "teal", "pair B"),
            highlight({4, 5}, "cyan", "pair C"),
            highlight({6, 7}, "blue", "pair D")
        },
        connectors = {
            connector(0, 2, "lime", "line"),
            connector(2, 4, "lime", "line"),
            connector(4, 6, "lime", "line")
        },
        message = "Doubles ladder " .. direction
    }
end
