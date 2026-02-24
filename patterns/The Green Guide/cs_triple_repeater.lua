--[[
Pattern: CS_TRIPLE_REPEATER
DisplayName: CS-Triple Repeater
Description: A 3-digit sequence repeats across the 8-digit serial as ABCABCAB. The sequence repeats twice fully (positions 1–6) and the first two digits appear again at positions 7–8. e.g., M 21021021 M.
BookRef: CS-1500
Tier: 7
Examples: ["21021021", "12312312", "01001001"]
Odds: 1 in 10,000
Price: $10-$100
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- ABCABCAB: pos 0=A, 1=B, 2=C, 3=A, 4=B, 5=C, 6=A, 7=B
    local a = d:sub(1, 1)
    local b = d:sub(2, 2)
    local c = d:sub(3, 3)

    if d:sub(4, 4) ~= a then return {matched = false} end
    if d:sub(5, 5) ~= b then return {matched = false} end
    if d:sub(6, 6) ~= c then return {matched = false} end
    if d:sub(7, 7) ~= a then return {matched = false} end
    if d:sub(8, 8) ~= b then return {matched = false} end

    -- Exclude CS-Solid (all same digit)
    if a == b and b == c then return {matched = false} end

    return {
        matched = true,
        group_boxes = {
            {from = 0, to = 2, color = "orange", thickness = 2},
            {from = 3, to = 5, color = "orange", thickness = 2},
            {from = 6, to = 7, color = "coral",  thickness = 2}
        },
        connectors = {
            {from = 0, to = 3, color = "orange", style = "arc"},
            {from = 2, to = 5, color = "orange", style = "arc"}
        },
        message = a .. b .. c .. " repeating (CS-Triple Repeater)"
    }
end
