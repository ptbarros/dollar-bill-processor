--[[
Pattern: SUM_0
Description: Minimum sum (all 0s)
Tier: 5
Examples: ["00000000"]
Odds: 1 in 100,000,000
Price: $500+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local sum = digit_sum(digits)
    if sum ~= 0 then
        return {matched = false}
    end

    -- Skip zeros: no box on any 0 digit (Ed review, applies to all Sum patterns).
    local positions = {}
    for i = 0, 7 do
        if digits:sub(i + 1, i + 1) ~= "0" then
            table.insert(positions, i)
        end
    end

    return {
        matched = true,
        highlights = {
            highlight(positions, "yellow", "min sum")
        },
        connectors = {},
        message = "Minimum digit sum = 0"
    }
end
