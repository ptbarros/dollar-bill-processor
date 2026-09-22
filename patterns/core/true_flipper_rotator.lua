--[[
Pattern: TRUE_FLIPPER_ROTATOR
DisplayName: True Flipper/Rotator
Description: Reads the SAME number upside down, using only 0, 6 and 9. A stricter cousin of the True Flipper: not just symmetric digits, but rotationally identical end-to-end.
Tier: 2
Examples: ["69069069", "66669999", "66696999"]
Odds: 1 in 1,548,387 (62 per 96M)
Price: $20-$100
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- First check all digits are flip-valid
    if not all_flip_valid(digits) then
        return {matched = false}
    end

    -- Get the flipped version
    local flipped = flip_string(digits)
    if not flipped then
        return {matched = false}
    end

    -- Rotator: reads the same when flipped
    if flipped ~= digits then
        return {matched = false}
    end

    -- Also check it only uses 0, 6, 9 (not 1 or 8 which flip to themselves)
    if not only_digits(digits, "069") then
        return {matched = false}
    end

    local positions = {0, 1, 2, 3, 4, 5, 6, 7}

    -- Add connectors showing the flip symmetry
    local connectors = {
        connector(0, 7, "purple", "arc"),
        connector(1, 6, "purple", "arc"),
        connector(2, 5, "purple", "arc"),
        connector(3, 4, "purple", "arc")
    }

    return {
        matched = true,
        highlights = {
            highlight(positions, "purple", "rotator")
        },
        connectors = connectors,
        message = "True flipper/rotator: reads the same number upside down"
    }
end
