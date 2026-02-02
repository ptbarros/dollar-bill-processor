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
