--[[
Pattern: LADDER_7
Description: Contains 7-digit ladder sequence
Tier: 2
Examples: ["12345670", "01234567", "76543210"]
Odds: 1 in 1,263,157
Price: $90-$500
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local result = find_ladder_of_length(digits, 7)
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
            highlight(positions, "lime", "ladder-7")
        },
        connectors = {},
        message = result.length .. "-digit " .. direction .. " ladder"
    }
end
