--[[
Pattern: TRIPLE_BOOKENDS
Description: First 3 and last 3 digits match
Tier: 4
Examples: ["12312312", "45645645"]
Odds: 1 in 1,000
Price: $5-$20
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local first3 = digits:sub(1, 3)
    local last3 = digits:sub(6, 8)

    if first3 ~= last3 then
        return {matched = false}
    end

    return {
        matched = true,
        highlights = {},
        connectors = {
            connector(1, 6, "orange", "arc")
        },
        group_boxes = {
            {from = 0, to = 2, color = "orange", thickness = 2},
            {from = 5, to = 7, color = "orange", thickness = 2}
        },
        message = "Triple bookends: " .. first3
    }
end
