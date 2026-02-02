--[[
Pattern: TRAILING_0000
Description: Ends with 0000
Tier: 9
Examples: ["12340000", "98760000"]
Odds: 1 in 10,000
Price: $5-$25
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    if not ends_with(digits, "0000") then
        return {matched = false}
    end

    -- Highlight the trailing zeros
    return {
        matched = true,
        highlights = {
            highlight({4, 5, 6, 7}, "gold", "trailing zeros")
        },
        connectors = {},
        message = "Ends with 0000"
    }
end
