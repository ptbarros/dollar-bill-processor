--[[
Pattern: NICKS_BROKEN_RADAR
DisplayName: Broken Radars
Description: Positions 1&8, 2&7, 3&6 match but 4&5 don't (near palindrome)
Tier: 5
Examples: ["12345621", "00112300", "98712389", "12003421"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Check outer pairs match
    if s:sub(1, 1) ~= s:sub(8, 8) then return {matched = false} end
    if s:sub(2, 2) ~= s:sub(7, 7) then return {matched = false} end
    if s:sub(3, 3) ~= s:sub(6, 6) then return {matched = false} end

    -- Middle pair should NOT match (otherwise it's a full radar)
    if s:sub(4, 4) == s:sub(5, 5) then
        return {matched = false}
    end

    return {
        matched = true,
        message = "Broken radar: outer 3 pairs match",
        highlights = {
            {positions = {0, 7}, color = "orange"},
            {positions = {1, 6}, color = "orange"},
            {positions = {2, 5}, color = "orange"},
            {positions = {3, 4}, color = "gray"}
        },
        connectors = {
            {from = 0, to = 7, color = "orange", style = "arc"},
            {from = 1, to = 6, color = "orange", style = "arc"},
            {from = 2, to = 5, color = "orange", style = "arc"}
        }
    }
end
