--[[
Pattern: FULL_HOUSE
Description: 5 of one digit, 3 of another
Tier: 4
Examples: ["11111222", "33333888", "55555333"]
Odds: 1 in 514
Price: $3-$8
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local counts = count_digits(digits)

    -- Find digits with count 5 and count 3
    local five_digit = nil
    local three_digit = nil

    for d, c in pairs(counts) do
        if c == 5 then
            five_digit = d
        elseif c == 3 then
            three_digit = d
        end
    end

    if not five_digit or not three_digit then
        return {matched = false}
    end

    local five_pos = find_digit_positions(digits, five_digit)
    local three_pos = find_digit_positions(digits, three_digit)

    return {
        matched = true,
        highlights = {
            highlight(five_pos, "gold", "five of kind"),
            highlight(three_pos, "coral", "three of kind")
        },
        connectors = {},
        message = "Full house: 5x" .. five_digit .. " + 3x" .. three_digit
    }
end
