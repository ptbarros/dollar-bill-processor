--[[
Pattern: CS_RANDOM_QUAD_AND_PAIRS
DisplayName: CS-Random Quad and Pairs
Description: Any CS-4OAK (four of one digit, scattered — no consecutive run of 4) plus any two CS-2OAKs (two different digits each appearing twice), all in any position. e.g., M 41142424 M.
BookRef: CS-320
Tier: 5
Examples: ["41142424", "24141424", "14421224"]
Odds: 1 in 2,160
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    local quad_digit = nil
    local pair_digits = {}

    for digit, cnt in pairs(counts) do
        if cnt == 4 then
            if quad_digit ~= nil then return {matched = false} end
            quad_digit = digit
        elseif cnt == 2 then
            table.insert(pair_digits, digit)
        else
            return {matched = false}
        end
    end

    -- Must have exactly one quad digit and exactly two pair digits (4+2+2=8)
    if not quad_digit or #pair_digits ~= 2 then
        return {matched = false}
    end

    -- The quad digit must NOT form a consecutive run of 4 (that would be CS-Quad and Pairs CS-310)
    for _, run in ipairs(find_runs(d)) do
        if run.digit == quad_digit and run.length >= 4 then
            return {matched = false}
        end
    end

    table.sort(pair_digits)
    local quad_pos = find_digit_positions(d, quad_digit)
    local pos_p1 = find_digit_positions(d, pair_digits[1])
    local pos_p2 = find_digit_positions(d, pair_digits[2])

    return {
        matched = true,
        highlights = {
            {positions = quad_pos, color = "gold"},
            {positions = pos_p1, color = "orange"},
            {positions = pos_p2, color = "coral"}
        },
        message = quad_digit .. "x4 scattered + pairs " .. pair_digits[1] .. "," .. pair_digits[2] .. " (CS-Random Quad and Pairs)"
    }
end
