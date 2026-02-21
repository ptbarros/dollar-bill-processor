--[[
Pattern: CS_TRUE_BINARY_FLIPPER
DisplayName: CS-True Binary Flipper
Description: Only 0s and 1s (CS-True Binary), AND reads the same upside-down (CS-Flipper). Subset of both CS-900 and CS-Rotator.
BookRef: CS-1040
Tier: 1
Examples: ["10100101", "11011011", "10111101"]
Price: $50-$500
--]]

function match(ctx)
    local d = ctx.digits

    -- Must contain only 0s and 1s
    if not only_digits(d, "01") then
        return {matched = false}
    end

    -- Must be a rotator: reads same upside-down
    -- For 0/1 only: flip_string reverses and maps 0→0, 1→1
    local flipped = flip_string(d)
    if flipped ~= d then
        return {matched = false}
    end

    local pos0 = find_digit_positions(d, "0")
    local pos1 = find_digit_positions(d, "1")

    return {
        matched = true,
        highlights = {
            {positions = pos0, color = "blue"},
            {positions = pos1, color = "cyan"},
        },
        message = "CS-True Binary Flipper: only 0/1, reads same upside-down (CS-1040)"
    }
end
