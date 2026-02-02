--[[
Pattern: LADDER
Description: Perfect 8-digit ascending or descending sequence
Tier: 1
Examples: ["01234567", "12345678", "87654321", "98765432"]
Odds: 1 in 19,200,000
Price: $500-$15,000
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local ascending = is_ascending(digits)
    local descending = is_descending(digits)

    if not ascending and not descending then
        return {matched = false}
    end

    local positions = {0, 1, 2, 3, 4, 5, 6, 7}
    local direction = ascending and "ascending" or "descending"

    -- Create connectors between consecutive digits
    local connectors = {}
    for i = 0, 6 do
        table.insert(connectors, connector(i, i + 1, "lime", "line"))
    end

    return {
        matched = true,
        highlights = {
            highlight(positions, "lime", "ladder")
        },
        connectors = connectors,
        message = "Perfect " .. direction .. " ladder"
    }
end
