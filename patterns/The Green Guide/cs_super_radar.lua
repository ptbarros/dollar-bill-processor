--[[
Pattern: CS_SUPER_RADAR
DisplayName: CS-Super Radar
Description: A CS-Sextup (6 consecutive identical digits) wrapped by a CS-Pair (same digit at both ends). Structure: ABBBBBBA where A ≠ B.
Tier: 2
Examples: ["04444440", "19999991", "30000003"]
Odds: 1 in 1,111,111
Price: $300-$1,500
--]]

function match(ctx)
    local d = ctx.digits

    -- Check: positions 0 and 7 are the same (the outer pair)
    local outer = d:sub(1, 1)
    if d:sub(8, 8) ~= outer then
        return {matched = false}
    end

    -- Check: positions 1-6 are all the same digit (the inner sextup)
    local inner = d:sub(2, 2)
    for i = 3, 7 do
        if d:sub(i, i) ~= inner then
            return {matched = false}
        end
    end

    -- Outer and inner digits must differ
    if outer == inner then
        return {matched = false}
    end

    return {
        matched = true,
        highlights = {
            {positions = {0, 7}, color = "coral"},
            {positions = {1, 2, 3, 4, 5, 6}, color = "gold"}
        },
        group_boxes = {
            {from = 1, to = 6, color = "gold", thickness = 3}
        },
        connectors = {
            {from = 0, to = 7, color = "coral", style = "arc"}
        },
        message = outer .. " wraps sextup of " .. inner .. "s (CS-Super Radar)"
    }
end
