--[[
Pattern: TRIPLE_BOOKEND
DisplayName: Triples Bookends
Description: First 3 digits match last 3 digits (ABCxxABC)
Tier: 4
Examples: ["12345123", "00112001", "98700987", "12300123"]
--]]

function match(ctx)
    local s = ctx.digits

    local first_three = s:sub(1, 3)
    local last_three = s:sub(6, 8)

    if first_three == last_three then
        return {
            matched = true,
            message = "Triple bookends: " .. first_three,
            group_boxes = {
                {from = 0, to = 2, color = "orange"},
                {from = 5, to = 7, color = "orange"}
            },
            connectors = {{from = 1, to = 6, color = "gold", style = "arc"}}
        }
    end

    return {matched = false}
end
