--[[
Pattern: STAR
Description: Star note (replacement bill)
Tier: 4
Examples: ["A12345678*"]
Odds: ~3% of print run
Price: Face value to $5+ (depends on other patterns)
--]]

function match(ctx)
    -- Star notes are identified by * suffix in the full serial
    local full_serial = ctx.full_serial
    if not full_serial then
        return {matched = false}
    end

    -- Check if it ends with *
    if not ends_with(full_serial, "*") then
        return {matched = false}
    end

    -- Highlight all digit positions (the star itself isn't in the digit positions)
    local positions = {0, 1, 2, 3, 4, 5, 6, 7}

    return {
        matched = true,
        highlights = {
            highlight(positions, "yellow", "star note")
        },
        connectors = {},
        message = "Star note (replacement bill)"
    }
end
