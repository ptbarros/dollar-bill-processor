--[[
Pattern: CS_PAIRED_30AK
DisplayName: CS-Paired 3OAK
Description: Two CS-Pairs and a CS-30AK (three of same digit scattered) anywhere in the serial, plus one random digit. e.g., M 1132233x M.
BookRef: CS-120
Tier: 8
Examples: ["11322334", "32231134", "11223312"]
Odds: 1 in 4,173,120
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    -- Must have exactly 4 distinct digits with counts {3, 2, 2, 1}
    local triple_digit, pair_digits, single_digit = nil, {}, nil
    for digit, cnt in pairs(counts) do
        if cnt == 3 then
            if triple_digit ~= nil then return {matched = false} end  -- two triples
            triple_digit = digit
        elseif cnt == 2 then
            table.insert(pair_digits, digit)
        elseif cnt == 1 then
            if single_digit ~= nil then return {matched = false} end  -- two singles
            single_digit = digit
        else
            return {matched = false}  -- count > 3 or == 0
        end
    end

    if triple_digit == nil or #pair_digits ~= 2 or single_digit == nil then
        return {matched = false}
    end

    -- Build highlights
    table.sort(pair_digits)
    local triple_positions = find_digit_positions(d, triple_digit)
    local pair1_positions = find_digit_positions(d, pair_digits[1])
    local pair2_positions = find_digit_positions(d, pair_digits[2])
    local single_positions = find_digit_positions(d, single_digit)

    return {
        matched = true,
        highlights = {
            {positions = triple_positions, color = "orange"},
            {positions = pair1_positions, color = "coral"},
            {positions = pair2_positions, color = "cyan"},
            {positions = single_positions, color = "gray"}
        },
        message = "30AK of " .. triple_digit .. " + pairs of " .. pair_digits[1] .. "," .. pair_digits[2] .. " (CS-Paired 30AK)"
    }
end
