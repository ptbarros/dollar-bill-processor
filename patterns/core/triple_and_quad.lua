--[[
Pattern: TRIPLE_AND_QUAD
Description: Triple + Quad combination
Tier: 5
Examples: ["11122223", "33334445", "00011112"]
Odds: 1 in 21,419 (4,482 per 96M)
Price: $20-$100+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Require consecutive runs: quad = run of 4+, triple = run of exactly 3
    local runs = find_runs(digits)
    local quad_run = nil
    local triple_run = nil

    for _, run in ipairs(runs) do
        if run.length == 4 and not quad_run then  -- exactly 4 (a 5+ run is N-of-a-kind, Ed review)
            quad_run = run
        elseif run.length == 3 and not triple_run then
            triple_run = run
        end
    end

    if not quad_run or not triple_run then
        return {matched = false}
    end

    local quad_pos = {}
    for i = quad_run.start, quad_run.start + quad_run.length - 1 do
        table.insert(quad_pos, i)
    end

    local triple_pos = {}
    for i = triple_run.start, triple_run.start + triple_run.length - 1 do
        table.insert(triple_pos, i)
    end

    return {
        matched = true,
        -- One box around each series, no per-digit boxes (Ed review).
        highlights = {},
        group_boxes = {
            {from = quad_pos[1], to = quad_pos[#quad_pos], color = "orange", thickness = 3},
            {from = triple_pos[1], to = triple_pos[#triple_pos], color = "magenta", thickness = 3}
        },
        connectors = {},
        message = "Triple + Quad: 3x" .. triple_run.digit .. " + 4x" .. quad_run.digit
    }
end
