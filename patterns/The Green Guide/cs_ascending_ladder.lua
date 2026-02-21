--[[
Pattern: CS_ASCENDING_LADDER
DisplayName: CS-Ascending Ladder
Description: All 8 digits form a consecutive ascending sequence (e.g., 12345678 or 23456789).
BookRef: CS-1170
Tier: 3
Examples: ["12345678", "23456789", "01234567"]
Odds: 1 in 6,944,444
Price: $1,000-$5,000
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then
        return {matched = false}
    end

    if not is_ascending(d) then
        return {matched = false}
    end

    local positions = {}
    for i = 0, 7 do
        table.insert(positions, i)
    end

    local connectors = {}
    for i = 0, 6 do
        table.insert(connectors, {from = i, to = i + 1, color = "lime", style = "line"})
    end

    return {
        matched = true,
        highlights = {{positions = positions, color = "lime"}},
        connectors = connectors,
        message = "Full 8-digit ascending ladder (CS-1170)"
    }
end
