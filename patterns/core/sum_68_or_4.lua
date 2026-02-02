--[[
Pattern: SUM_68_OR_4
Description: Digit sum equals 68 or 4
Tier: 5
Examples: ["99999992", "00000004"]
Odds: 1 in 200,000
Price: $40-$300+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local sum = digit_sum(digits)
    if sum ~= 68 and sum ~= 4 then
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
