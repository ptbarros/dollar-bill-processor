--[[
Pattern: SUM_64_OR_8
Description: Digit sum equals 64 or 8
Tier: 5
Examples: ["99999996", "00000008"]
Odds: 1 in 7,692
Price: $10-$50+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local sum = digit_sum(digits)
    if sum ~= 64 and sum ~= 8 then
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
