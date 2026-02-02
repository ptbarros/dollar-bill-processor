--[[
Pattern: SUM_66_OR_6
Description: Digit sum equals 66 or 6
Tier: 5
Examples: ["99999994", "00000006"]
Odds: 1 in 33,333
Price: $10-$100
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local sum = digit_sum(digits)
    if sum ~= 66 and sum ~= 6 then
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
