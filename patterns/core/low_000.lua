--[[
Pattern: LOW_000
Description: Exactly three leading zeros — no fewer, no more (e.g. 000·12345).
Tier: 4
Examples: ["00012345", "00098765"]
Odds: 1 in 1,067 (90,000 per 96M)
Price: $5-$35
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Exactly 3 leading zeros: starts with 000 but the 4th digit is not a zero.
    if not starts_with(digits, "000") or digits:sub(4, 4) == "0" then
        return {matched = false}
    end

    return {
        matched = true,
        highlights = {},
        group_boxes = {},
        connectors = {},
        message = "Low serial (3 leading zeros)"
    }
end
