--[[
Pattern: TRUE_FLIPPER_ROTATOR
DisplayName: True Flipper/Rotator
Description: Reads the SAME number upside down, using only 0, 6 and 9. A stricter cousin of the True Flipper: not just symmetric digits, but rotationally identical end-to-end.
Tier: 2
Flippable: true
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

    -- Two-tone halves so the rotational symmetry reads at a glance: the front
    -- half and back half are different colours, and each arc links a front digit
    -- to the back digit it becomes when the note is turned 180°. (When the overlay
    -- can also be drawn flipped, the two colours land on the opposite side.)
    local connectors = {
        connector(0, 7, "magenta", "arc"),
        connector(1, 6, "magenta", "arc"),
        connector(2, 5, "magenta", "arc"),
        connector(3, 4, "magenta", "arc")
    }

    return {
        matched = true,
        highlights = {
            highlight({0, 1, 2, 3}, "blue", "front half"),
            highlight({4, 5, 6, 7}, "orange", "back half")
        },
        connectors = connectors,
        message = "True flipper/rotator: reads the same number upside down"
    }
end
