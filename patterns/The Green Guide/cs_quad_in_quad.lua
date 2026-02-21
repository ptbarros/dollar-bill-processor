--[[
Pattern: CS_QUAD_IN_QUAD
DisplayName: CS-Quad in Quad
Description: A CS-Quad (4 consecutive) grouped within a CS-40AK (4 scattered of another digit). Exactly two digits, each 4 times, one as a 4-run, one scattered. e.g., M 54444555 M or M 55444455 M. Also a CS-Binary.
BookRef: CS-250
Tier: 4
Examples: ["54444555", "55444455", "55544445"]
Odds: 1 in 90
Price: $25-$500
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    -- Must be exactly 2 distinct digits, each with count 4
    local digits_list = {}
    for digit, cnt in pairs(counts) do
        if cnt ~= 4 then return {matched = false} end
        table.insert(digits_list, digit)
    end
    if #digits_list ~= 2 then return {matched = false} end

    local d1, d2 = digits_list[1], digits_list[2]

    -- Determine which has the 4-run (CS-Quad) and which is scattered (CS-40AK)
    local runs = find_runs(d)
    local function max_run_for(digit)
        local mx = 0
        for _, run in ipairs(runs) do
            if run.digit == digit and run.length > mx then mx = run.length end
        end
        return mx
    end

    local d1_max = max_run_for(d1)
    local d2_max = max_run_for(d2)

    local quad_digit, scatter_digit, quad_run
    if d1_max >= 4 and d2_max < 4 then
        quad_digit = d1
        scatter_digit = d2
    elseif d2_max >= 4 and d1_max < 4 then
        quad_digit = d2
        scatter_digit = d1
    else
        return {matched = false}  -- neither or both have 4-run
    end

    -- Find the quad run for highlighting
    for _, run in ipairs(runs) do
        if run.digit == quad_digit and run.length >= 4 then
            quad_run = run
            break
        end
    end

    local scatter_positions = find_digit_positions(d, scatter_digit)

    return {
        matched = true,
        group_boxes = {
            {from = quad_run.start, to = quad_run.start + quad_run.length - 1, color = "gold", thickness = 3}
        },
        highlights = {
            {positions = scatter_positions, color = "coral"}
        },
        message = quad_digit .. "x4 quad inside " .. scatter_digit .. "x4 scattered (CS-Quad in Quad)"
    }
end
