--[[
Pattern: SUM_72
Description: Maximum sum (all 9s)
Tier: 5
Examples: ["99999999"]
Odds: 1 in 100,000,000
Price: $500+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local sum = digit_sum(digits)
    if sum ~= 72 then
        return {matched = false}
    end

    local positions = {0, 1, 2, 3, 4, 5, 6, 7}

    return {
        matched = true,
        highlights = {
            highlight(positions, "yellow", "max sum")
        },
        connectors = {},
        message = "Maximum digit sum = 72"
    }
end
