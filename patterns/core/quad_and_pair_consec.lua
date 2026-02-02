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

    -- Look for a consecutive pair adjacent to the quad or elsewhere
    local runs = find_runs(digits)
    local has_pair = false
    local pair_pos = nil

    for _, run in ipairs(runs) do
        if run.length == 2 and run.digit ~= quad.digit then
            has_pair = true
            pair_pos = run.start
            break
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
        highlights = {
            highlight(quad_positions, "gold", "quad"),
            highlight({pair_pos, pair_pos + 1}, "coral", "pair")
        },
        group_boxes = {
            {from = quad.start, to = quad.start + quad.length - 1, color = "gold", thickness = 2},
            {from = pair_pos, to = pair_pos + 1, color = "coral", thickness = 2}
        },
        connectors = {},
        message = "Quad " .. quad.digit .. "s + consecutive pair"
    }
end
