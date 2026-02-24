--[[
Pattern: CS_DUAL_RADAR_BOOKEND
DisplayName: CS-Dual Radar Bookend
Description: The first two digits are mirrored (reversed) at the end of the serial: A B xxxx B A, where A ≠ B. Distinguished from CS-Dual Repeater Bookend (CS-980) where the sequence repeats in the same order (A B xxxx A B). e.g., M 23xxxx32 M.
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
