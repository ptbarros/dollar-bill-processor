--[[
Pattern: CS_RADAR_REPEATER
DisplayName: CS-Radar Repeater
Description: A CS-Full Repeater (first half = second half: ABCDABCD) that is also a CS-Full Radar (palindrome). e.g., 12211221.
BookRef: CS-1520
Tier: 3
Examples: ["12211221", "34433443", "10011001"]
Price: $25-$500
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Must be a CS-Full Repeater: first half = second half
    if not is_repeater(d) then
        return {matched = false}
    end

    -- Must also be a palindrome (CS-Full Radar)
    if not is_palindrome(d) then
        return {matched = false}
    end

    local half = d:sub(1, 4)

    return {
        matched = true,
        group_boxes = {
            {from = 0, to = 3, color = "magenta", thickness = 2},
            {from = 4, to = 7, color = "magenta", thickness = 2},
        },
        connectors = {
            {from = 0, to = 7, color = "orange", style = "arc"},
            {from = 1, to = 6, color = "orange", style = "arc"},
            {from = 0, to = 4, color = "magenta", style = "arc"},
        },
        message = half .. " repeats AND is a palindrome (CS-Radar Repeater CS-1520)"
    }
end
