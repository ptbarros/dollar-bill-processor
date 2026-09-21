--[[
Pattern: SUM_69_OR_3
Description: Digit sum equals 69 or 3
Tier: 5
Examples: ["00000003", "00000030", "00000300"]
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
