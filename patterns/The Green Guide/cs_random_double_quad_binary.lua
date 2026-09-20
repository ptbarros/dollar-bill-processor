--[[
Pattern: CS_RANDOM_DOUBLE_QUAD_BINARY
DisplayName: Binary Quads
Description: Only two digits, each used four times and mixed together rather than split into two clean blocks (e.g. 4111·4414).
BookRef: CS-930
Tier: 5
Examples: ["41114414", "11441144", "41414141"]
Odds: 1 in 17,500
Price: $5-$20
--]]

function match(ctx)
    local d = ctx.digits

    -- Must use exactly 2 distinct digits
    if unique_count(d) ~= 2 then
        return {matched = false}
    end

    -- Each digit must appear exactly 4 times.
    local counts = count_digits(d)
    for _, cnt in pairs(counts) do
        if cnt ~= 4 then return {matched = false} end
    end
    -- Collect the two digits in reading order (first appearance) so colours are
    -- deterministic (pairs() order is randomised per process).
    local digit_list = {}
    local seen = {}
    for i = 1, 8 do
        local ch = d:sub(i, i)
        if not seen[ch] then seen[ch] = true; table.insert(digit_list, ch) end
    end

    -- Exclude if both quads are consecutive (00001111 or 11110000) — that's CS-920
    if d:sub(1,4) == d:sub(1,1):rep(4) and d:sub(5,8) == d:sub(5,5):rep(4) then
        return {matched = false}
    end

    local d1 = digit_list[1]
    local d2 = digit_list[2]
    local pos1 = find_digit_positions(d, d1)
    local pos2 = find_digit_positions(d, d2)

    return {
        matched = true,
        highlights = {
            {positions = pos1, color = "gold"},
            {positions = pos2, color = "coral"}
        },
        message = "Random double quad binary: four " .. d1 .. "s and four " .. d2 .. "s (CS-Random Double Quad Binary)"
    }
end
