--[[
Pattern: CS_PAIRS_IN_QUAD
DisplayName: CS-Pairs in Quad
Description: A CS-4OAK bookending the serial (first and last digit both the quad digit), plus two CS-Pairs inside. Exactly 3 distinct digits: one with count 4, two with count 2. e.g., M 44114224 M.
BookRef: CS-330
Tier: 4
Examples: ["44114224", "99118899", "55335885"]
Odds: 1 in 140,000
Price: $10-$30
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- First and last digit must match (the quad digit bookend)
    local quad_digit = d:sub(1, 1)
    if d:sub(8, 8) ~= quad_digit then
        return {matched = false}
    end

    -- Count all digits
    local counts = count_digits(d)

    -- The quad digit must appear exactly 4 times
    if (counts[quad_digit] or 0) ~= 4 then
        return {matched = false}
    end

    -- Must have exactly 3 distinct digits total
    local num_distinct = 0
    for _, _ in pairs(counts) do
        num_distinct = num_distinct + 1
    end
    if num_distinct ~= 3 then
        return {matched = false}
    end

    -- The other two digits must each appear exactly 2 times
    local pair_digits = {}
    for digit, cnt in pairs(counts) do
        if digit ~= quad_digit then
            if cnt ~= 2 then
                return {matched = false}
            end
            table.insert(pair_digits, digit)
        end
    end

    -- Find positions for visualization
    local quad_pos = find_digit_positions(d, quad_digit)
    local pair1_pos = find_digit_positions(d, pair_digits[1])
    local pair2_pos = find_digit_positions(d, pair_digits[2])

    return {
        matched = true,
        highlights = {
            {positions = quad_pos, color = "gold"},
            {positions = pair1_pos, color = "orange"},
            {positions = pair2_pos, color = "coral"}
        },
        connectors = {
            {from = 0, to = 7, color = "gold", style = "arc"}
        },
        message = "Pairs in quad: four " .. quad_digit .. "s bookend two pairs of " .. pair_digits[1] .. " and " .. pair_digits[2] .. " (CS-Pairs in Quad)"
    }
end
