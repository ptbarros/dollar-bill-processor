--[[
Pattern: LOW_000000
Description: Starts with 000000 (serial under 100)
Tier: 2
Examples: ["00000012", "00000099"]
Odds: 1 in 1,066,667
Price: $100-$800
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    if not starts_with(digits, "000000") then
        return {matched = false}
    end

    -- Highlight the leading zeros
    return {
        matched = true,
        highlights = {
            highlight({0, 1, 2, 3, 4, 5}, "gold", "ultra low serial")
        },
        connectors = {},
        message = "Ultra low serial (under 100)"
    }
end
