--[[
Pattern: SUM_67_OR_5
Description: Digit sum equals 67 or 5
Tier: 5
Examples: ["99999993", "00000005"]
Odds: 1 in 76,923
Price: $20-$140+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local sum = digit_sum(digits)
    if sum ~= 67 and sum ~= 5 then
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
