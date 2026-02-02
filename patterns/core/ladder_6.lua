--[[
Pattern: LADDER_6
Description: Contains 6-digit ladder
Tier: 3
Examples: ["12345622", "34567822", "65432100"]
Odds: 1 in 71,428
Price: $30-$350
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local result = find_ladder_of_length(digits, 6)
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
            highlight(positions, "lime", "ladder-6")
        },
        connectors = {},
        message = result.length .. "-digit " .. direction .. " ladder"
    }
end
