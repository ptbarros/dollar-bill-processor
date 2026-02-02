--[[
Pattern: ALL_ODDS
Description: All digits odd (1,3,5,7,9)
Tier: 4
Examples: ["13579135", "97531975", "13579997"]
Odds: 1 in 256
Price: $3-$10
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    if not only_digits(digits, "13579") then
        return {matched = false}
    end

    -- Highlight all positions
    local positions = {0, 1, 2, 3, 4, 5, 6, 7}

    return {
        matched = true,
        highlights = {
            highlight(positions, "orange", "odd")
        },
        connectors = {},
        message = "All odd digits"
    }
end
