--[[
Pattern: SUM_70_OR_2
Description: Digit sum equals 70 or 2
Tier: 5
Examples: ["99999990", "00000002"]
Odds: 1 in 2,232,558
Price: $100-$1,000+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local sum = digit_sum(digits)
    if sum ~= 70 and sum ~= 2 then
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
