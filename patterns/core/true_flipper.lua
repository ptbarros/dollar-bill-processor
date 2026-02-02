--[[
Pattern: TRUE_FLIPPER
Description: Reads same upside down (only 0, 6, 9)
Tier: 3
Examples: ["69000069", "96099069", "00699600"]
Odds: 1 in 18,443
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

    -- True flipper: reads the same when flipped
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
            highlight(positions, "purple", "true flipper")
        },
        connectors = connectors,
        message = "True flipper: reads same upside down"
    }
end
