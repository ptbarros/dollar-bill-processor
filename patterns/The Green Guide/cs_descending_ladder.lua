--[[
Pattern: CS_DESCENDING_LADDER
DisplayName: CS-Descending Ladder
Description: All 8 digits form a consecutive descending sequence (e.g., 87654321 or 98765432).
BookRef: CS-1180
Tier: 3
Examples: ["87654321", "98765432", "76543210"]
Odds: 1 in 6,944,444
Price: $1,000-$5,000
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then
        return {matched = false}
    end

    if not is_descending(d) then
        return {matched = false}
    end

    local positions = {}
    for i = 0, 7 do
        table.insert(positions, i)
    end

    local connectors = {}
    for i = 0, 6 do
        table.insert(connectors, {from = i, to = i + 1, color = "coral", style = "line"})
    end

    return {
        matched = true,
        highlights = {{positions = positions, color = "coral"}},
        connectors = connectors,
        message = "Full 8-digit descending ladder (CS-1180)"
    }
end
