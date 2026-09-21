--[[
Pattern: LOW_0000
Description: Exactly four leading zeros — no fewer, no more (e.g. 0000·1234).
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

    -- Exactly 4 leading zeros: starts with 0000 but the 5th digit is not a zero.
    if not starts_with(digits, "0000") or digits:sub(5, 5) == "0" then
        return {matched = false}
    end

    return {
        matched = true,
        highlights = {},
        group_boxes = {},
        connectors = {},
        message = "Low serial (4 leading zeros)"
    }
end
