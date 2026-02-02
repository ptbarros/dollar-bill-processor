--[[
Pattern: DOUBLE_FOUR_OF_KIND
Description: Two groups of 4 of a kind (33343444)
Tier: 3
Examples: ["33343444", "11121222", "55545666"]
Odds: 1 in 31,250
Price: $20-$120
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local counts = count_digits(digits)

    -- Find two digits each with count 4
    local four_digits = {}
    for d, c in pairs(counts) do
        if c == 4 then
            table.insert(four_digits, d)
        end
    end

    if #four_digits ~= 2 then
        return {matched = false}
    end

    local pos1 = find_digit_positions(digits, four_digits[1])
    local pos2 = find_digit_positions(digits, four_digits[2])

    return {
        matched = true,
        highlights = {
            highlight(pos1, "gold", "first 4"),
            highlight(pos2, "coral", "second 4")
        },
        connectors = {},
        message = "Double 4-of-a-kind: 4x" .. four_digits[1] .. " + 4x" .. four_digits[2]
    }
end
