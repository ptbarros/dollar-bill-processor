--[[
Pattern: TRUE_FLIPPER
DisplayName: True Flipper
Description: Only the perfectly-symmetric flip digits 0, 6 and 9. Always also a Flipper, but 1 and 8 are excluded because they don't truly look the same upside down. Reads as a valid number when flipped (it just need not be the SAME number -- that's a Rotator).
Tier: 5
Flippable: true
Examples: ["06960690", "96069600", "60909690"]
Odds: 1 in 18,812 (5,103 per 96M)
Price: $5-$25
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Purists' flipper: only the digits that look the same flipped (0, 6, 9).
    -- 1 and 8 are flip-valid but not perfectly symmetric, so they're excluded.
    if not only_digits(digits, "069") then
        return {matched = false}
    end

    -- Color-code each digit by how it transforms, matching Flipper: 0 flips to
    -- itself (purple); 6 and 9 flip INTO each other, so give them distinct
    -- colours (6 magenta, 9 orange) -- that way the swap is visible, and in the
    -- 180-degree flipped view each colour lands where its partner digit now sits.
    local highlights = {}
    for i = 0, 7 do
        local d = digits:sub(i + 1, i + 1)
        local color
        if d == "0" then
            color = "purple"
        elseif d == "6" then
            color = "magenta"
        else  -- d == "9"
            color = "orange"
        end
        table.insert(highlights, highlight({i}, color, d))
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = {},
        message = "True flipper: only perfectly-symmetric digits (0, 6, 9)"
    }
end
