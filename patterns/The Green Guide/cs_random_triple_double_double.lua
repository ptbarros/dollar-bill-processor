--[[
Pattern: CS_RANDOM_TRIPLE_DOUBLE_DOUBLE
DisplayName: CS-Random Triple Double Double
Description: One 3OAK (one digit appearing 3 times) and two 2OAKs (two different digits each appearing twice), in any position (scattered). e.g., M 333221x1 M. They can be a CS-Triple and CS-Pairs as long as one of the three is an OAK.
BookRef: CS-180
Tier: 6
Examples: ["11312244", "33322141", "44421312"]
Odds: 1 in 3,360
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    local triple_digit = nil
    local pair_digits = {}
    local single_digit = nil

    for digit, cnt in pairs(counts) do
        if cnt == 3 then
            if triple_digit ~= nil then return {matched = false} end
            triple_digit = digit
        elseif cnt == 2 then
            table.insert(pair_digits, digit)
        elseif cnt == 1 then
            if single_digit ~= nil then return {matched = false} end
            single_digit = digit
        else
            return {matched = false}
        end
    end

    -- Must have exactly 1 triple + 2 pairs + 1 single = 3+2+2+1 = 8
    if not triple_digit or #pair_digits ~= 2 or not single_digit then
        return {matched = false}
    end

    table.sort(pair_digits)
    local pos_triple = find_digit_positions(d, triple_digit)
    local pos_p1 = find_digit_positions(d, pair_digits[1])
    local pos_p2 = find_digit_positions(d, pair_digits[2])
    local pos_single = find_digit_positions(d, single_digit)

    -- If ALL three groups (triple + 2 pairs) are consecutive, this is CS-170 (Triple Double Double)
    local function all_consec(positions)
        return positions[#positions] - positions[1] == #positions - 1
    end
    if all_consec(pos_triple) and all_consec(pos_p1) and all_consec(pos_p2) then
        return {matched = false}
    end

    return {
        matched = true,
        highlights = {
            {positions = pos_triple, color = "gold"},
            {positions = pos_p1, color = "orange"},
            {positions = pos_p2, color = "coral"},
            -- the one leftover digit is NOT part of the pattern: mark it with an X
            -- rather than a muted box (a box reads like a faint match)
            {positions = pos_single, color = "charcoal", style = "x"}
        },
        message = triple_digit .. "x3 + " .. pair_digits[1] .. "x2 + " .. pair_digits[2] .. "x2 + " .. single_digit .. " (CS-Random Triple Double Double)"
    }
end
