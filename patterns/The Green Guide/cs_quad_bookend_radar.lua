--[[
Pattern: CS_QUAD_BOOKEND_RADAR
DisplayName: Quad Bookend Radar
Description: A pair of one digit at each end wrapping a solid block of four of another digit (e.g. 22·4444·22).
BookRef: CS-1310
Tier: 2
Examples: ["22444422", "33666633", "11999911"]
Odds: 1 in 1,185,185 (81 per 96M)
Price: $500-$3,000
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Structure: AABBBBAA
    -- Outer pair: positions 0-1 and 6-7 are the same digit (A)
    -- Inner quad: positions 2-5 are the same digit (B)
    -- A ~= B

    local outer = d:sub(1, 1)
    if d:sub(2, 2) ~= outer then return {matched = false} end
    if d:sub(7, 7) ~= outer then return {matched = false} end
    if d:sub(8, 8) ~= outer then return {matched = false} end

    local inner = d:sub(3, 3)
    if inner == outer then return {matched = false} end
    for i = 4, 6 do
        if d:sub(i, i) ~= inner then return {matched = false} end
    end

    return {
        matched = true,
        group_boxes = {
            {from = 0, to = 1, color = "coral", thickness = 3},
            {from = 2, to = 5, color = "gold", thickness = 3},
            {from = 6, to = 7, color = "coral", thickness = 3}
        },
        connectors = {},  -- arcs removed (Ed review)
        message = outer .. outer .. " wraps quad of " .. inner .. "s (CS-Quad Bookend Radar)"
    }
end
