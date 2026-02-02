--[[
Pattern: LADDER_5
Description: Contains 5-digit ladder
Tier: 4
Examples: ["12345999", "63567893", "98765123"]
Odds: 1 in 4,348
Price: $4-$20
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local result = find_ladder_of_length(digits, 5)
    if not result then
        return {matched = false}
    end

    -- Highlight the ladder
    local positions = {}
    for i = 0, result.length - 1 do
        table.insert(positions, result.start + i)
    end

    local direction = result.ascending and "ascending" or "descending"

    return {
        matched = true,
        highlights = {
            highlight(positions, "lime", "ladder-5")
        },
        connectors = {},
        message = result.length .. "-digit " .. direction .. " ladder"
    }
end
