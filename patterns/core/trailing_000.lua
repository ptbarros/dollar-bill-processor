--[[
Pattern: TRAILING_000
Description: Ends with 000
Tier: 9
Examples: ["12345000", "98765000"]
Odds: 1 in 1,000
Price: $3-$10
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    if not ends_with(digits, "000") then
        return {matched = false}
    end

    -- Highlight the trailing zeros
    return {
        matched = true,
        highlights = {
            highlight({5, 6, 7}, "gold", "trailing zeros")
        },
        connectors = {},
        message = "Ends with 000"
    }
end
