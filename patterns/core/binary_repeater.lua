--[[
Pattern: BINARY_REPEATER
Description: Binary AND repeater
Tier: 2
Examples: ["01010101", "10101010", "33993399"]
Odds: 1 in 1,185,185
Price: $80-$450
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check binary (exactly 2 unique digits)
    if unique_count(digits) ~= 2 then
        return {matched = false}
    end

    -- Check repeater (ABCDABCD)
    if not is_repeater(digits) then
        return {matched = false}
    end

    -- Get the two digits
    local unique = get_unique_digits(digits)
    local d1, d2 = unique:sub(1, 1), unique:sub(2, 2)

    return {
        matched = true,
        highlights = {
            highlight(find_digit_positions(digits, d1), "blue", "digit 1"),
            highlight(find_digit_positions(digits, d2), "cyan", "digit 2")
        },
        connectors = {
            connector(0, 4, "magenta", "arc"),
            connector(1, 5, "magenta", "arc"),
            connector(2, 6, "magenta", "arc"),
            connector(3, 7, "magenta", "arc")
        },
        message = "Binary (" .. d1 .. "," .. d2 .. ") + Repeater"
    }
end
