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

    local positions = {0, 1, 2, 3, 4, 5, 6, 7}

    return {
        matched = true,
        highlights = {
            highlight(positions, "yellow", "min sum")
        },
        connectors = {},
        message = "Minimum digit sum = 0"
    }
end
