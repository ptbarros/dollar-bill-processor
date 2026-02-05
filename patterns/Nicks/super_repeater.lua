--[[
Pattern: NICKS_SUPER_REPEATER
DisplayName: Super Repeater
Description: 2-digit pattern repeated 4 times (XYXYXYXY)
Tier: 2
Examples: ["12121212", "01010101", "98989898", "12121212"]
--]]

function match(ctx)
    local s = ctx.digits

    local pair = s:sub(1, 2)
    local expected = pair .. pair .. pair .. pair

    if s == expected then
        return {
            matched = true,
            message = "Super repeater: " .. pair .. " × 4",
            group_boxes = {
                {from = 0, to = 1, color = "gold"},
                {from = 2, to = 3, color = "gold"},
                {from = 4, to = 5, color = "gold"},
                {from = 6, to = 7, color = "gold"}
            }
        }
    end

    return {matched = false}
end
