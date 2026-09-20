--[[
Pattern: CS_DUAL_RADAR_BOOKEND
DisplayName: 2 Digit Radar Bookend
Description: The first two digits show up flipped at the tail end — two different digits mirrored across the serial (e.g. 23·xxxx·32).
BookRef: CS-970
Tier: 7
Examples: ["23000032", "45678954", "12999921"]
Odds: 1 in 810,000
Price: $10-$100
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    local a = d:sub(1, 1)  -- position 0 (1-indexed: 1)
    local b = d:sub(2, 2)  -- position 1 (1-indexed: 2)

    -- A and B must differ
    if a == b then return {matched = false} end

    -- Mirrored at end: position 6 = B, position 7 = A (0-indexed)
    if d:sub(7, 7) ~= b then return {matched = false} end
    if d:sub(8, 8) ~= a then return {matched = false} end

    return {
        matched = true,
        group_boxes = {
            {from = 0, to = 1, color = "orange", thickness = 3},
            {from = 6, to = 7, color = "orange", thickness = 3}
        },
        connectors = {
            {from = 0, to = 7, color = "orange", style = "arc"},
            {from = 1, to = 6, color = "coral",  style = "arc"}
        },
        message = a .. b .. " mirrored as " .. b .. a .. " at ends (CS-Dual Radar Bookend)"
    }
end
