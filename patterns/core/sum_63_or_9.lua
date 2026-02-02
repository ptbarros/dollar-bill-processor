--[[
Pattern: SUM_63_OR_9
Description: Digit sum equals 63 or 9
Tier: 5
Examples: ["99999997", "00000009"]
Odds: 1 in 4,166
Price: $10-$40+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local sum = digit_sum(digits)
    if sum ~= 63 and sum ~= 9 then
        return {matched = false}
    end

    local positions = {0, 1, 2, 3, 4, 5, 6, 7}

    return {
        matched = true,
        highlights = {
            highlight(positions, "purple", "sum")
        },
        connectors = {},
        message = "Digit sum = " .. sum
    }
end
