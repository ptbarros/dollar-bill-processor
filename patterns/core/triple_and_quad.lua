--[[
Pattern: TRIPLE_AND_QUAD
Description: Triple + Quad combination
Tier: 3
Examples: ["11122222", "33334445", "00011111"]
Odds: 1 in 24,691
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
        if run.length >= 4 and not quad_run then
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
        highlights = {
            highlight(quad_pos, "gold", "quad"),
            highlight(triple_pos, "coral", "triple")
        },
        connectors = {},
        message = "Triple + Quad: 3x" .. triple_run.digit .. " + " .. quad_run.length .. "x" .. quad_run.digit
    }
end
