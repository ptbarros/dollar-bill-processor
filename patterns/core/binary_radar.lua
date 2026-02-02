--[[
Pattern: BINARY_RADAR
Description: Binary AND palindrome
Tier: 2
Examples: ["10011001", "66166166", "01100110"]
Odds: 1 in 914,285
Price: $80-$4,500
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

    -- Check palindrome
    if not is_palindrome(digits) then
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
            connector(0, 7, "orange", "arc"),
            connector(1, 6, "orange", "arc"),
            connector(2, 5, "orange", "arc"),
            connector(3, 4, "orange", "arc")
        },
        message = "Binary (" .. d1 .. "," .. d2 .. ") + Radar"
    }
end
