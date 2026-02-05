--[[
Pattern: NICKS_RADAR
DisplayName: Radar
Description: Palindrome - reads same forwards and backwards
Tier: 4
Examples: ["12344321", "00011000", "98766789", "12300321"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Check palindrome
    for i = 1, 4 do
        if s:sub(i, i) ~= s:sub(9 - i, 9 - i) then
            return {matched = false}
        end
    end

    return {
        matched = true,
        message = "Radar (palindrome)",
        highlights = {{positions = {0,1,2,3,4,5,6,7}, color = "orange"}},
        connectors = {
            {from = 0, to = 7, color = "orange", style = "arc"},
            {from = 1, to = 6, color = "orange", style = "arc"},
            {from = 2, to = 5, color = "orange", style = "arc"},
            {from = 3, to = 4, color = "orange", style = "arc"}
        }
    }
end
