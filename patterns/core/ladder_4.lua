--[[
Pattern: LADDER_4
Description: Contains 4-digit ladder
Tier: 4
Examples: ["12349876", "01234999", "98765111"]
Odds: 1 in 303
Price: $3-$10
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local result = find_ladder_of_length(digits, 4)
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
            highlight(positions, "lime", "ladder-4")
        },
        connectors = {},
        message = result.length .. "-digit " .. direction .. " ladder"
    }
end
