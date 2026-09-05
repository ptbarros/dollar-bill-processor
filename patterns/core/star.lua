--[[
Pattern: STAR
Description: Star note (replacement bill)
Tier: 4
Overlay: none
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

    -- A star note's fanciness is the * suffix, not the digits, so draw nothing on
    -- the digits -- boxing every digit was just noise. (STAR is also hidden from
    -- the overlay picker via "Overlay: none"; this keeps the crop clean even when
    -- STAR is forced via right-click "Set Pattern...".)
    return {
        matched = true,
        highlights = {},
        connectors = {},
        message = "Star note (replacement bill)"
    }
end
