--[[
Pattern: SUM_67_OR_5
Description: Digit sum equals 67 or 5
Tier: 4
Examples: ["04000001", "00001400", "00000500"]
Odds: 1 in 85,031 (1,129 per 96M)
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
