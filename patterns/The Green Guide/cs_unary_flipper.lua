--[[
Pattern: CS_UNARY_FLIPPER
DisplayName: CS-Unary Flipper
Description: Every digit is the same, and it's one that still reads upside-down — a solid note of all 1s, 6s, 8s or 9s (e.g. 88888888).
BookRef: CS-1030
Tier: 1
Examples: ["11111111", "66666666", "88888888", "99999999"]
Odds: 1 in 11,111,111
Price: $1,000+
--]]

function match(ctx)
    local d = ctx.digits
    local first = d:sub(1, 1)

    -- Must be a valid flipper digit — not 0 (00000000 is CS-Solid but not Unary Flipper)
    if first ~= "1" and first ~= "6" and first ~= "8" and first ~= "9" then
        return {matched = false}
    end

    -- All 8 digits must be the same
    for i = 2, 8 do
        if d:sub(i, i) ~= first then return {matched = false} end
    end

    return {
        matched = true,
        group_boxes = {
            {from = 0, to = 7, color = "purple", thickness = 3}
        },
        message = "All " .. first .. "s — CS-Unary Flipper (CS-Solid + flip)"
    }
end
