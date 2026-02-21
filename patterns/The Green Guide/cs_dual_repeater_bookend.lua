--[[
Pattern: CS_DUAL_REPEATER_BOOKEND
DisplayName: CS-Dual Repeater Bookend
Description: First two digits repeat at the end in the same order (e.g., 12xxxx12). Both bookend digits must be different — if they are the same (11...11) that is CS-960.
BookRef: CS-980
Tier: 7
Examples: ["12000012", "23456723", "34567834"]
Odds: 1 in 900,000
Price: $10-$100
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- First 2 digits must match last 2 digits in same order
    if not is_bookended(d, 2) then
        return {matched = false}
    end

    local b1 = d:sub(1, 1)
    local b2 = d:sub(2, 2)

    -- Both bookend digits must differ — same digit (AA...AA) is CS-960
    if b1 == b2 then
        return {matched = false}
    end

    return {
        matched = true,
        group_boxes = {
            {from = 0, to = 1, color = "orange", thickness = 3},
            {from = 6, to = 7, color = "orange", thickness = 3}
        },
        connectors = {
            {from = 0, to = 6, color = "orange", style = "arc"},
            {from = 1, to = 7, color = "coral",  style = "arc"}
        },
        message = b1 .. b2 .. " repeated at both ends (CS-Dual Repeater Bookend)"
    }
end
