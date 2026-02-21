--[[
Pattern: CS_QUAD_BOOKEND_RADAR
DisplayName: CS-Quad Bookend Radar
Description: A CS-40AK split as a pair at each end, bookending an inner CS-Quad. Structure AABBBBAA. e.g., M 22444422 M or M 33666633 M.
BookRef: CS-1310
Tier: 2
Examples: ["22444422", "33666633", "11999911"]
Odds: 1 in 1,111,111
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
        connectors = {
            {from = 0, to = 7, color = "coral", style = "arc"},
            {from = 1, to = 6, color = "coral", style = "arc"}
        },
        message = outer .. outer .. " wraps quad of " .. inner .. "s (CS-Quad Bookend Radar)"
    }
end
