--[[
Pattern: RADAR_REPEATER
Description: Both radar AND repeater (e.g., 12121212)
Tier: 2
Examples: ["12121212", "59955995", "34433443"]
Odds: 1 in 1,111,111
Price: $80-$450
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check palindrome (radar)
    if not is_palindrome(digits) then
        return {matched = false}
    end

    -- Check repeater (ABCDABCD)
    if not is_repeater(digits) then
        return {matched = false}
    end

    return {
        matched = true,
        highlights = {
            highlight({0, 1, 2, 3}, "orange", "first half"),
            highlight({4, 5, 6, 7}, "magenta", "second half")
        },
        connectors = {
            connector(0, 7, "orange", "arc"),
            connector(1, 6, "orange", "arc"),
            connector(2, 5, "orange", "arc"),
            connector(3, 4, "orange", "arc")
        },
        message = "Radar + Repeater"
    }
end
