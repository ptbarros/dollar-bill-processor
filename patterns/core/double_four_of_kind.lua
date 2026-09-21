--[[
Pattern: DOUBLE_FOUR_OF_KIND
DisplayName: Double Four of a Kind
Description: Two groups of 4 of a kind (33343444)
Tier: 3
Examples: ["33343444", "11121222", "55556666"]
Odds: 1 in 32,487 (2,955 per 96M)
Price: $20-$120
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local counts = count_digits(digits)

    -- Find the two digits each with count 4, in reading order (first appearance) so
    -- their colours are deterministic (pairs() order is randomised per process).
    local four_digits = {}
    local seen = {}
    for i = 1, 8 do
        local d = digits:sub(i, i)
        if not seen[d] then
            seen[d] = true
            if counts[d] == 4 then table.insert(four_digits, d) end
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
