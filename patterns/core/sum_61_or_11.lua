--[[
Pattern: SUM_61_OR_11
Description: Digit sum equals 61 or 11
Tier: 5
Examples: ["89999999", "20000009"]
Odds: 1 in 1,315
Price: $5-$20
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local sum = digit_sum(digits)
    if sum ~= 61 and sum ~= 11 then
        return {matched = false}
    end

    -- Highlight all digits
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
