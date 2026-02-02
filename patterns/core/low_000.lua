--[[
Pattern: LOW_000
Description: Starts with 000
Tier: 4
Examples: ["00012345", "00098765"]
Odds: 1 in 1,066
Price: $5-$35
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    if not starts_with(digits, "000") then
        return {matched = false}
    end

    -- Highlight the leading zeros
    return {
        matched = true,
        highlights = {
            highlight({0, 1, 2}, "gold", "low serial")
        },
        connectors = {},
        message = "Low serial (starts with 000)"
    }
end
