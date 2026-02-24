--[[
Pattern: CS_RANDOM_TRIPLE_TRIPLE_PAIR
DisplayName: CS-Random Triple Triple Pair
Description: Two 3OAKs (two different digits each appearing 3 times) and one 2OAK (a third digit appearing twice), in any position (scattered). e.g., M 12332112 M. These are also a CS-Trinary.
BookRef: CS-140
Tier: 5
Examples: ["12332112", "33344452", "11122355"]
Odds: 1 in 5,040
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    local triple_digits = {}
    local pair_digit = nil
    local pair_count = 0

    for digit, cnt in pairs(counts) do
        if cnt == 3 then
            table.insert(triple_digits, digit)
        elseif cnt == 2 then
            pair_digit = digit
            pair_count = pair_count + 1
        elseif cnt ~= 0 then
            return {matched = false}
        end
    end

    -- Must have exactly 2 triple digits and exactly 1 pair digit
    if #triple_digits ~= 2 or pair_count ~= 1 then
        return {matched = false}
    end

    -- 2+3+3 = 8 digits total — valid
    table.sort(triple_digits)
    local pos_t1 = find_digit_positions(d, triple_digits[1])
    local pos_t2 = find_digit_positions(d, triple_digits[2])
    local pos_pair = find_digit_positions(d, pair_digit)

    -- If ALL three groups are consecutive, this is CS-130 (Triple Triple Pair), not CS-140
    local function all_consec(positions)
        return positions[#positions] - positions[1] == #positions - 1
    end
    if all_consec(pos_t1) and all_consec(pos_t2) and all_consec(pos_pair) then
        return {matched = false}
    end

    return {
        matched = true,
        highlights = {
            {positions = pos_t1, color = "gold"},
            {positions = pos_t2, color = "orange"},
            {positions = pos_pair, color = "coral"}
        },
        message = triple_digits[1] .. "x3 + " .. triple_digits[2] .. "x3 + " .. pair_digit .. "x2 (CS-Random Triple Triple Pair)"
    }
end
