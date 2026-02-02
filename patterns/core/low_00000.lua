--[[
Pattern: LOW_00000
Description: Starts with 00000 (serial under 1000)
Tier: 3
Examples: ["00000123", "00000999"]
Odds: 1 in 106,666
Price: $40-$200
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    if not starts_with(digits, "00000") then
        return {matched = false}
    end

    -- Highlight the leading zeros
    return {
        matched = true,
        highlights = {
            highlight({0, 1, 2, 3, 4}, "gold", "low serial")
        },
        connectors = {},
        message = "Very low serial (under 1000)"
    }
end
