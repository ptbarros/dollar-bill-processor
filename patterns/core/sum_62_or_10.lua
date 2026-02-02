--[[
Pattern: SUM_62_OR_10
Description: Digit sum equals 62 or 10
Tier: 5
Examples: ["99999998", "10000009"]
Odds: 1 in 2,272
Price: $5-$25+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local sum = digit_sum(digits)
    if sum ~= 62 and sum ~= 10 then
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
