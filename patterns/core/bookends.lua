--[[
Pattern: BOOKENDS
Description: First 2 and last 2 digits match
Tier: 4
Examples: ["12345612", "99123499"]
Odds: 1 in 100
Price: $3-$10
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local first2 = digits:sub(1, 2)
    local last2 = digits:sub(7, 8)

    if first2 ~= last2 then
        return {matched = false}
    end

    return {
        matched = true,
        highlights = {},
        connectors = {
            connector(0, 7, "orange", "arc")
        },
        group_boxes = {
            {from = 0, to = 1, color = "orange", thickness = 2},
            {from = 6, to = 7, color = "orange", thickness = 2}
        },
        message = "Bookends: " .. first2
    }
end
