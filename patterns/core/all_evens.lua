--[[
Pattern: ALL_EVENS
Description: All digits even (0,2,4,6,8)
Tier: 4
Examples: ["24680246", "20486420", "24688864"]
Odds: 1 in 256
Price: $3-$10
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    if not only_digits(digits, "02468") then
        return {matched = false}
    end

    -- Highlight all positions
    local positions = {0, 1, 2, 3, 4, 5, 6, 7}

    return {
        matched = true,
        highlights = {
            highlight(positions, "blue", "even")
        },
        connectors = {},
        message = "All even digits"
    }
end
