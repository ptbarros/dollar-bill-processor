--[[
Pattern: NICKS_REPEATER
DisplayName: Repeater
Description: First 4 digits repeat in positions 5-8 (ABCDABCD)
Tier: 4
Examples: ["12341234", "00110011", "98769876", "12001200"]
--]]

function match(ctx)
    local s = ctx.digits

    local first_half = s:sub(1, 4)
    local second_half = s:sub(5, 8)

    if first_half == second_half then
        return {
            matched = true,
            message = "Repeater: " .. first_half .. " × 2",
            group_boxes = {
                {from = 0, to = 3, color = "orange"},
                {from = 4, to = 7, color = "orange"}
            },
            connectors = {{from = 1, to = 5, color = "gold", style = "arc"}}
        }
    end

    return {matched = false}
end
