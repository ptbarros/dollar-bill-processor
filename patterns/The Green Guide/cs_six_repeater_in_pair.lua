--[[
Pattern: CS_SIX_REPEATER_IN_PAIR
DisplayName: CS-Six Repeater in a Pair
Description: A 3-digit sequence repeated twice (6 digits) is bookended by a CS-2OAK: A · XYZ · XYZ · A, where A appears at positions 1 and 8 and differs from all of X, Y, Z. e.g., M 9 301 301 9 M.
BookRef: CS-1510
Tier: 6
Examples: ["93013019", "15151510", "87878780"]
Odds: 1 in 10,000
Price: $10-$100
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    local outer = d:sub(1, 1)  -- position 0

    -- Outer digits must match: pos 0 = pos 7
    if d:sub(8, 8) ~= outer then return {matched = false} end

    -- Middle 6 digits (pos 1–6) must form a 3-digit repeater: XYZ XYZ
    local x = d:sub(2, 2)
    local y = d:sub(3, 3)
    local z = d:sub(4, 4)

    if d:sub(5, 5) ~= x then return {matched = false} end
    if d:sub(6, 6) ~= y then return {matched = false} end
    if d:sub(7, 7) ~= z then return {matched = false} end

    -- Outer digit must not appear in the repeating sequence (it's a distinct CS-2OAK)
    if outer == x or outer == y or outer == z then return {matched = false} end

    -- The XYZ pattern must not be all the same digit
    if x == y and y == z then return {matched = false} end

    return {
        matched = true,
        highlights = {
            {positions = {0, 7}, color = "cyan"}
        },
        group_boxes = {
            {from = 1, to = 3, color = "orange", thickness = 2},
            {from = 4, to = 6, color = "orange", thickness = 2}
        },
        connectors = {
            {from = 0, to = 7, color = "cyan",   style = "arc"},
            {from = 1, to = 4, color = "orange", style = "arc"},
            {from = 3, to = 6, color = "orange", style = "arc"}
        },
        message = outer .. " bookends " .. x .. y .. z .. " repeated twice (CS-Six Repeater in a Pair)"
    }
end
