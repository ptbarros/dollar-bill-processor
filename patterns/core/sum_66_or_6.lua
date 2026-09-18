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

    -- Skip zeros: no box on any 0 digit (Ed review, applies to all Sum patterns).
    local positions = {}
    for i = 0, 7 do
        if digits:sub(i + 1, i + 1) ~= "0" then
            table.insert(positions, i)
        end
    end

    return {
        matched = true,
        highlights = {
            highlight(positions, "purple", "sum")
        },
        connectors = {},
        message = "Digit sum = " .. sum
    }
end
