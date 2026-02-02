--[[
Pattern: SUM_65_OR_7
Description: Digit sum equals 65 or 7
Tier: 5
Examples: ["99999995", "00000007"]
Odds: 1 in 15,625
Price: $5-$100+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local sum = digit_sum(digits)
    if sum ~= 65 and sum ~= 7 then
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
