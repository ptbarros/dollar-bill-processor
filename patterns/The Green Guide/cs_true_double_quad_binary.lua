--[[
Pattern: CS_TRUE_DOUBLE_QUAD_BINARY
DisplayName: CS-True Double Quad Binary
Description: Two consecutive quads of 0s and 1s. Only two possible serials: 00001111 or 11110000.
BookRef: CS-920
Tier: 2
Examples: ["00001111", "11110000"]
Odds: 1 in 50,000,000
Price: $50+
--]]

function match(ctx)
    local d = ctx.digits

    if d ~= "00001111" and d ~= "11110000" then
        return {matched = false}
    end

    local first_color, second_color
    if d == "00001111" then
        first_color = "blue"
        second_color = "gold"
    else
        first_color = "gold"
        second_color = "blue"
    end

    return {
        matched = true,
        group_boxes = {
            {from = 0, to = 3, color = first_color, thickness = 3},
            {from = 4, to = 7, color = second_color, thickness = 3}
        },
        connectors = {
            {from = 0, to = 7, color = "purple", style = "arc"}
        },
        message = "True double quad binary: " .. d:sub(1,4) .. " | " .. d:sub(5,8) .. " (CS-True Double Quad Binary)"
    }
end
