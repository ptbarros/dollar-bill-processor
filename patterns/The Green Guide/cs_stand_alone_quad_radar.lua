--[[
Pattern: CS_STAND_ALONE_QUAD_RADAR
DisplayName: CS-Stand Alone Quad Radar
Description: A CS-Radar surrounded by zeros. 6-digit palindrome at positions 1-6 (ABCCBA or ABBBBA), with zeros at positions 0 and 7. e.g., M 01233210 M or M 01444410 M.
BookRef: CS-1770
Tier: 3
Examples: ["01233210", "01444410", "02133120"]
Odds: 1 in 27
Price: $0-$5
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Positions 0 and 7 must be zero
    if d:sub(1, 1) ~= "0" or d:sub(8, 8) ~= "0" then
        return {matched = false}
    end

    -- 6-digit palindrome at positions 1-6 (1-indexed: 2-7)
    -- d[2]==d[7], d[3]==d[6], d[4]==d[5]
    local p1 = d:sub(2, 2)
    local p2 = d:sub(3, 3)
    local p3 = d:sub(4, 4)
    local p4 = d:sub(5, 5)
    local p5 = d:sub(6, 6)
    local p6 = d:sub(7, 7)

    if p1 == p6 and p2 == p5 and p3 == p4 and p1 ~= "0" then
        return {
            matched = true,
            group_boxes = {
                {from = 1, to = 6, color = "orange", thickness = 3}
            },
            connectors = {
                {from = 1, to = 6, color = "orange", style = "arc"},
                {from = 2, to = 5, color = "coral", style = "arc"},
                {from = 3, to = 4, color = "gold", style = "arc"}
            },
            message = p1 .. p2 .. p3 .. p4 .. p5 .. p6 .. " stand-alone quad-radar (CS-Stand Alone Quad Radar)"
        }
    end

    return {matched = false}
end
