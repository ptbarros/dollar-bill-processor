--[[
Pattern: CS_QUAD_AND_PAIRS
DisplayName: CS-Quad and Pairs
Description: A CS-Quad (4 consecutive same digit) plus two CS-Pairs of different digits, using all 8 positions. e.g., M 11222233 M or M 44441122 M.
BookRef: CS-310
Tier: 5
Examples: ["11222233", "44441122", "33224444"]
Odds: 1 in 2,160
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    -- Must have exactly 3 distinct digits with counts {4, 2, 2}
    local quad_digit, pair_digits = nil, {}
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

    if not quad_digit or #pair_digits ~= 2 then
        return {matched = false}
    end

    -- Quad digit must form a consecutive 4-run
    local quad_run = nil
    for _, run in ipairs(find_runs(d)) do
        if run.digit == quad_digit and run.length >= 4 then
            quad_run = run
            break
        end
    end
    if not quad_run then return {matched = false} end

    table.sort(pair_digits)
    local pair1_pos = find_digit_positions(d, pair_digits[1])
    local pair2_pos = find_digit_positions(d, pair_digits[2])

    return {
        matched = true,
        group_boxes = {
            {from = quad_run.start, to = quad_run.start + quad_run.length - 1, color = "gold", thickness = 3}
        },
        highlights = {
            {positions = pair1_pos, color = "orange"},
            {positions = pair2_pos, color = "coral"}
        },
        message = quad_digit .. "x4 quad + pairs of " .. pair_digits[1] .. "," .. pair_digits[2] .. " (CS-Quad and Pairs)"
    }
end
