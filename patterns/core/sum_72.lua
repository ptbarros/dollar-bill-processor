--[[
Pattern: SUM_72
Description: Maximum sum (all 9s)
Tier: 5
Examples: ["99999999"]
Odds: Cannot occur on a printed note — only 99999999, which is never printed
Price: $500+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local sum = digit_sum(digits)
    if sum ~= 72 then
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
            highlight(positions, "yellow", "max sum")
        },
        connectors = {},
        message = "Maximum digit sum = 72"
    }
end
