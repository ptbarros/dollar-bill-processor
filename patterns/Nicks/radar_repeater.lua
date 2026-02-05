--[[
Pattern: NICKS_RADAR_REPEATER
DisplayName: Radar Repeater
Description: Both a radar (palindrome) and repeater (ABBAABBA)
Tier: 2
Examples: ["12211221", "00110011", "98899889", "12001200"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Check repeater
    local first_half = s:sub(1, 4)
    local second_half = s:sub(5, 8)
    if first_half ~= second_half then
        return {matched = false}
    end

    -- Check palindrome
    for i = 1, 4 do
        if s:sub(i, i) ~= s:sub(9 - i, 9 - i) then
            return {matched = false}
        end
    end

    return {
        matched = true,
        message = "Radar repeater: " .. first_half .. " × 2 (palindrome)",
        group_boxes = {
            {from = 0, to = 3, color = "gold"},
            {from = 4, to = 7, color = "gold"}
        },
        connectors = {
            {from = 0, to = 7, color = "orange", style = "arc"},
            {from = 1, to = 6, color = "orange", style = "arc"}
        }
    }
end
