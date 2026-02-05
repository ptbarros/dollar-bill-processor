--[[
Pattern: DOUBLE_BOOKEND
DisplayName: Doubles Bookends
Description: First 2 digits match last 2 digits (ABxxxxAB)
Tier: 5
Examples: ["12345612", "00112300", "98700098", "12000012"]
--]]

function match(ctx)
    local s = ctx.digits

    -- First 2 and last 2 must match
    local first_two = s:sub(1, 2)
    local last_two = s:sub(7, 8)

    if first_two ~= last_two then
        return {matched = false}
    end

    -- But not triple bookend (first 3 = last 3)
    local first_three = s:sub(1, 3)
    local last_three = s:sub(6, 8)
    if first_three == last_three then
        return {matched = false}
    end

    -- And not quad bookend (first 4 = last 4)
    local first_four = s:sub(1, 4)
    local last_four = s:sub(5, 8)
    if first_four == last_four then
        return {matched = false}
    end

    return {
        matched = true,
        message = "Double bookends: " .. first_two,
        group_boxes = {
            {from = 0, to = 1, color = "orange"},
            {from = 6, to = 7, color = "orange"}
        },
        connectors = {{from = 0, to = 6, color = "gold", style = "arc"}}
    }
end
