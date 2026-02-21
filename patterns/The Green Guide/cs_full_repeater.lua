--[[
Pattern: CS_FULL_REPEATER
DisplayName: CS-Paired Quad Repeater
Description: The first four digits are repeated exactly in positions 5-8 (ABCDABCD pattern).
BookRef: CS-1480
Tier: 3
Examples: ["12341234", "56785678", "00120012"]
Odds: 1 in 10,000
Price: $25-$150
--]]

function match(ctx)
    local d = ctx.digits
    if not is_repeater(d) then
        return {matched = false}
    end

    local half = d:sub(1, 4)
    local pair_colors = {"orange", "coral", "cyan", "lime"}
    local group_boxes = {}

    -- Each position i matches position i+4
    for i = 0, 3 do
        local color = pair_colors[(i % #pair_colors) + 1]
        table.insert(group_boxes, {from = i, to = i, color = color, thickness = 2})
        table.insert(group_boxes, {from = i + 4, to = i + 4, color = color, thickness = 2})
    end

    return {
        matched = true,
        group_boxes = {
            {from = 0, to = 3, color = "magenta", thickness = 2},
            {from = 4, to = 7, color = "magenta", thickness = 2}
        },
        connectors = {
            {from = 0, to = 4, color = "magenta", style = "arc"},
            {from = 3, to = 7, color = "magenta", style = "arc"}
        },
        message = half .. " repeats (CS-Paired Quad Repeater)"
    }
end
