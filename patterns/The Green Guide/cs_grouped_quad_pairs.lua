--[[
Pattern: CS_GROUPED_QUAD_PAIRS
DisplayName: CS-Quad Pairs
Description: Four consecutive pairs (AABBCCDD) filling all 8 positions, with all four pair digits distinct. e.g., M 11223344 M or M 99887766 M.
BookRef: CS-60
Tier: 5
Examples: ["11223344", "99887766", "00112233"]
Odds: 1 in 50,000
Price: $20-$100
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- All four consecutive pairs must exist
    if not has_four_consecutive_pairs(d) then
        return {matched = false}
    end

    -- All four pair digits must be distinct (AABBCCDD, not AABBAACC)
    local p = {}
    for i = 1, 4 do
        p[i] = d:sub((i - 1) * 2 + 1, (i - 1) * 2 + 1)
    end
    for i = 1, 3 do
        for j = i + 1, 4 do
            if p[i] == p[j] then
                return {matched = false}
            end
        end
    end

    local pair_colors = {"orange", "coral", "cyan", "lime"}
    local highlights = {}
    for i = 1, 4 do
        local base = (i - 1) * 2
        table.insert(highlights, {positions = {base, base + 1}, color = pair_colors[i]})
    end

    return {
        matched = true,
        highlights = highlights,
        group_boxes = {
            {from = 0, to = 7, color = "gold", thickness = 3}
        },
        message = p[1]..p[1]..p[2]..p[2]..p[3]..p[3]..p[4]..p[4] .. " — four grouped pairs (CS-Grouped Quad Pairs)"
    }
end
