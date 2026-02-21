--[[
Pattern: CS_QUAD_IN_TRIPLE
DisplayName: CS-Quad in Triple
Description: A CS-Quad (4 consecutive) within a CS-30AK (3 of another digit surrounding it), plus one random digit. e.g., M x3344443 M or M 3444433x M.
BookRef: CS-260
Tier: 6
Examples: ["13444433", "34444331", "03444432"]
Odds: 1 in 720
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    -- Must have exactly 3 distinct digits with counts {4, 3, 1}
    local quad_digit, triple_digit, single_digit = nil, nil, nil
    for digit, cnt in pairs(counts) do
        if cnt == 4 then
            if quad_digit ~= nil then return {matched = false} end
            quad_digit = digit
        elseif cnt == 3 then
            if triple_digit ~= nil then return {matched = false} end
            triple_digit = digit
        elseif cnt == 1 then
            if single_digit ~= nil then return {matched = false} end
            single_digit = digit
        else
            return {matched = false}
        end
    end

    if not quad_digit or not triple_digit or not single_digit then
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

    -- Triple digit must appear on BOTH sides of the quad run (surrounding it)
    local triple_positions = find_digit_positions(d, triple_digit)
    local has_before, has_after = false, false
    for _, pos in ipairs(triple_positions) do
        if pos < quad_run.start then has_before = true end
        if pos >= quad_run.start + quad_run.length then has_after = true end
    end
    if not has_before or not has_after then return {matched = false} end

    local single_positions = find_digit_positions(d, single_digit)

    return {
        matched = true,
        group_boxes = {
            {from = quad_run.start, to = quad_run.start + quad_run.length - 1, color = "gold", thickness = 3}
        },
        highlights = {
            {positions = triple_positions, color = "orange"},
            {positions = single_positions, color = "gray"}
        },
        message = quad_digit .. "x4 quad within " .. triple_digit .. "x3 triple (CS-Quad in Triple)"
    }
end
