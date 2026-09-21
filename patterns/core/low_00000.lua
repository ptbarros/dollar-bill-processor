--[[
Pattern: LOW_00000
Description: Exactly five leading zeros, so the serial is under 1000 (e.g. 00000·123).
Tier: 4
Examples: ["00000123", "00000999"]
Odds: 1 in 106,667 (900 per 96M)
Price: $40-$200
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Exactly 5 leading zeros: starts with 00000 but the 6th digit is not a zero.
    -- (On an 8-digit serial this also guarantees the number is under 1000.)
    if not starts_with(digits, "00000") or digits:sub(6, 6) == "0" then
        return {matched = false}
    end

    return {
        matched = true,
        highlights = {},
        group_boxes = {},
        connectors = {},
        message = "Very low serial (under 1000)"
    }
end
