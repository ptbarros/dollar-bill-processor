--[[
Pattern: LOW_0000
Description: Starts with 0000
Tier: 4
Examples: ["00001234", "00009999"]
Odds: 1 in 10,667
Price: $10-$30
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    if not starts_with(digits, "0000") then
        return {matched = false}
    end

    -- Highlight the leading zeros
    return {
        matched = true,
        highlights = {
            highlight({0, 1, 2, 3}, "gold", "low serial")
        },
        connectors = {},
        message = "Low serial (starts with 0000)"
    }
end
