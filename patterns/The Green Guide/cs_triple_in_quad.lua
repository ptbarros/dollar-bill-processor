--[[
Pattern: CS_TRIPLE_IN_QUAD
DisplayName: CS-Triple in Quad
Description: A CS-Triple (3 consecutive) within a CS-40AK (4 scattered of another digit surrounding it), plus one random digit. e.g., M 4443334x M.
BookRef: CS-290
Tier: 5
Examples: ["44433340", "14443334", "04443334"]
Odds: 1 in 12,960
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    -- Must have exactly 3 distinct digits with counts {3, 4, 1}
    local triple_digit, quad_digit, single_digit = nil, nil, nil
    for digit, cnt in pairs(counts) do
        if cnt == 3 then
            if triple_digit ~= nil then return {matched = false} end
            triple_digit = digit
        elseif cnt == 4 then
            if quad_digit ~= nil then return {matched = false} end
            quad_digit = digit
        elseif cnt == 1 then
            if single_digit ~= nil then return {matched = false} end
            single_digit = digit
        else
            return {matched = false}
        end
    end

    if not triple_digit or not quad_digit or not single_digit then
        return {matched = false}
    end

    -- Triple digit must form a consecutive 3-run
    local triple_run = nil
    for _, run in ipairs(find_runs(d)) do
        if run.digit == triple_digit and run.length >= 3 then
            triple_run = run
            break
        end
    end
    if not triple_run then return {matched = false} end

    -- Quad digit must NOT have a 4-run (must be scattered = CS-40AK)
    for _, run in ipairs(find_runs(d)) do
        if run.digit == quad_digit and run.length >= 4 then
            return {matched = false}
        end
    end

    -- Quad digit must appear on BOTH sides of the triple run (surrounding it)
    local quad_positions = find_digit_positions(d, quad_digit)
    local has_before, has_after = false, false
    for _, pos in ipairs(quad_positions) do
        if pos < triple_run.start then has_before = true end
        if pos >= triple_run.start + triple_run.length then has_after = true end
    end
    if not has_before or not has_after then return {matched = false} end

    local single_positions = find_digit_positions(d, single_digit)

    return {
        matched = true,
        group_boxes = {
            {from = triple_run.start, to = triple_run.start + triple_run.length - 1, color = "orange", thickness = 3}
        },
        highlights = {
            {positions = quad_positions, color = "gold"},
            {positions = single_positions, color = "charcoal", style = "x"}
        },
        message = triple_digit .. "x3 triple within " .. quad_digit .. "x4 scattered (CS-Triple in Quad)"
    }
end
