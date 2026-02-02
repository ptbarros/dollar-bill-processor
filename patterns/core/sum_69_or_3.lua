--[[
Pattern: SUM_69_OR_3
Description: Digit sum equals 69 or 3
Tier: 5
Examples: ["99999991", "00000003"]
Odds: 1 in 588,957
Price: $50-$500+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local sum = digit_sum(digits)
    if sum ~= 69 and sum ~= 3 then
        return {matched = false}
    end

    local positions = {0, 1, 2, 3, 4, 5, 6, 7}

    return {
        matched = true,
        highlights = {
            highlight(positions, "gold", "sum")
        },
        connectors = {},
        message = "Digit sum = " .. sum
    }
end
