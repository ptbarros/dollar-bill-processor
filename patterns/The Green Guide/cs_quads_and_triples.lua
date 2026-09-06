--[[
Pattern: CS_QUADS_AND_TRIPLES
DisplayName: CS-Quads and Triples
Description: A CS-Quad (4 consecutive same digit) and a CS-Triple (3 consecutive same digit), where the remaining digit cannot match either the quad or triple digit. e.g., M 4444333x M, M 4444x333 M, M x4444333 M.
BookRef: CS-340
Tier: 4
Examples: ["44443330", "33344441", "14444333"]
Odds: 1 in 2,520
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    local runs = find_runs(d)

    local quad_run, triple_run = nil, nil

    for _, run in ipairs(runs) do
        if run.length >= 4 and not quad_run then
            quad_run = run
        elseif run.length >= 3 and run.length < 4 and not triple_run then
            triple_run = run
        elseif run.length >= 3 then
            -- Second run of 3+ would be another triple — allow if it's the quad
            if run.length >= 4 and not quad_run then
                quad_run = run
            end
        end
    end

    if not quad_run or not triple_run then
        return {matched = false}
    end

    -- Quad and triple must be different digits
    if quad_run.digit == triple_run.digit then
        return {matched = false}
    end

    -- Total covered positions
    local total = quad_run.length + triple_run.length
    if total > 8 then
        return {matched = false}
    end

    -- Remaining digits (if any) must not match quad or triple digit
    local remaining = 8 - quad_run.length - triple_run.length
    if remaining > 0 then
        for i = 0, 7 do
            local in_quad = i >= quad_run.start and i < quad_run.start + quad_run.length
            local in_triple = i >= triple_run.start and i < triple_run.start + triple_run.length
            if not in_quad and not in_triple then
                local ch = d:sub(i + 1, i + 1)
                if ch == quad_run.digit or ch == triple_run.digit then
                    return {matched = false}
                end
            end
        end
    end

    -- Collect remaining positions for highlighting
    local remaining_pos = {}
    for i = 0, 7 do
        local in_quad = i >= quad_run.start and i < quad_run.start + quad_run.length
        local in_triple = i >= triple_run.start and i < triple_run.start + triple_run.length
        if not in_quad and not in_triple then
            table.insert(remaining_pos, i)
        end
    end

    local group_boxes = {
        {from = quad_run.start, to = quad_run.start + quad_run.length - 1, color = "gold", thickness = 3},
        {from = triple_run.start, to = triple_run.start + triple_run.length - 1, color = "orange", thickness = 3}
    }

    local highlights = {}
    if #remaining_pos > 0 then
        table.insert(highlights, {positions = remaining_pos, color = "charcoal", style = "x"})
    end

    return {
        matched = true,
        group_boxes = group_boxes,
        highlights = highlights,
        message = quad_run.digit .. "x" .. quad_run.length .. " + " .. triple_run.digit .. "x" .. triple_run.length .. " (CS-Quads and Triples)"
    }
end
