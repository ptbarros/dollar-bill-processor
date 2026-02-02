--[[
Pattern: CONSECUTIVE_TRIPLES
Description: Two triples back-to-back (AAACCCXX)
Tier: 4
Examples: ["11122234", "33344456", "22233345"]
Odds: 1 in 4,938
Price: $10-$30+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Find triples
    local triples = find_triples(digits)

    if #triples < 2 then
        return {matched = false}
    end

    -- Check if first two triples are consecutive in position
    local t1 = triples[1]
    local t2 = triples[2]

    -- They should be adjacent (t1 ends where t2 starts)
    if t1.start + t1.length ~= t2.start then
        return {matched = false}
    end

    local t1_positions = {}
    for i = 0, t1.length - 1 do
        table.insert(t1_positions, t1.start + i)
    end

    local t2_positions = {}
    for i = 0, t2.length - 1 do
        table.insert(t2_positions, t2.start + i)
    end

    return {
        matched = true,
        highlights = {
            highlight(t1_positions, "gold", "triple 1"),
            highlight(t2_positions, "coral", "triple 2")
        },
        group_boxes = {
            {from = t1.start, to = t1.start + t1.length - 1, color = "gold", thickness = 2},
            {from = t2.start, to = t2.start + t2.length - 1, color = "coral", thickness = 2}
        },
        connectors = {},
        message = "Consecutive triples: " .. t1.digit .. t1.digit .. t1.digit .. " + " .. t2.digit .. t2.digit .. t2.digit
    }
end
