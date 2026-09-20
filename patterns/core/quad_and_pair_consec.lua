--[[
Pattern: QUAD_AND_PAIR_CONSEC
Description: 4 in a row plus consecutive pair
Tier: 4
Examples: ["11112234", "22223345"]
Odds: 1 in 5,643
Price: $5-$25
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Find a quad (4+ consecutive identical)
    local quad = has_n_consecutive(digits, 4)
    if not quad then
        return {matched = false}
    end

    -- Look for a consecutive pair ADJACENT to the quad (Ed review): the pair must
    -- start immediately after the quad ends, or end immediately before it begins,
    -- so the two boxes read as one block. Pick by adjacency, not by position.
    local runs = find_runs(digits)
    local has_pair = false
    local pair_pos = nil
    local quad_end = quad.start + quad.length - 1  -- 0-indexed

    for _, run in ipairs(runs) do
        if run.length == 2 and run.digit ~= quad.digit then
            if run.start == quad_end + 1 or run.start + 2 == quad.start then
                has_pair = true
                pair_pos = run.start
                break
            end
        end
    end

    if not has_pair then
        return {matched = false}
    end

    local quad_positions = {}
    for i = 0, quad.length - 1 do
        table.insert(quad_positions, quad.start + i)
    end

    return {
        matched = true,
        -- Inner per-digit boxes removed (Ed review); keep the chunk group boxes.
        highlights = {},
        group_boxes = {
            {from = quad.start, to = quad.start + quad.length - 1, color = "gold", thickness = 2},
            {from = pair_pos, to = pair_pos + 1, color = "coral", thickness = 2}
        },
        connectors = {},
        message = "Quad " .. quad.digit .. "s + consecutive pair"
    }
end
