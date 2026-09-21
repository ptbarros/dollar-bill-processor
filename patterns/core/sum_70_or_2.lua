--[[
Pattern: SUM_70_OR_2
Description: Digit sum equals 70 or 2
Tier: 5
Examples: ["00002000", "00010100", "00101000"]
Odds: 1 in 2,181,818 (44 per 96M)
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
            highlight(positions, "gold", "sum")
        },
        connectors = {},
        message = "Digit sum = " .. sum
    }
end
