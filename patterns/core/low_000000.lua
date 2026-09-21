--[[
Pattern: LOW_000000
Description: Exactly six leading zeros, so the serial is under 100 (e.g. 000000·12).
Tier: 2
Examples: ["00000012", "00000099"]
Odds: 1 in 1,066,667 (90 per 96M)
Price: $100-$800
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Exactly 6 leading zeros: starts with 000000 but the 7th digit is not a zero.
    -- (On an 8-digit serial this also guarantees the number is under 100.)
    if not starts_with(digits, "000000") or digits:sub(7, 7) == "0" then
        return {matched = false}
    end

    return {
        matched = true,
        highlights = {},
        group_boxes = {},
        connectors = {},
        message = "Ultra low serial (under 100)"
    }
end
