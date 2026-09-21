--[[
Pattern: CS_DESCENDING_LADDER
DisplayName: CS-Descending Ladder
Description: Every digit drops by one straight across the serial, rolling from 0 back to 9 if it needs to (e.g. 87654321, 09876543).
BookRef: CS-1180
Tier: 3
Examples: ["87654321", "09876543", "76543210"]
Odds: 1 in 6,944,444
Price: $1,000-$5,000
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then
        return {matched = false}
    end

    -- Allow mod-10 wrap: each digit must be (first - offset) % 10
    local first = tonumber(d:sub(1, 1))
    for i = 2, 8 do
        if tonumber(d:sub(i, i)) ~= (first - (i - 1) + 100) % 10 then
            return {matched = false}
        end
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
