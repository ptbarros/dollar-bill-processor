--[[
Pattern: CS_SUPER_REPEATER
DisplayName: CS-Super Repeater
Description: Two different digits alternate four times (ABABABAB pattern, A≠B).
BookRef: CS-1530
Tier: 2
Examples: ["12121212", "34343434", "01010101"]
Odds: 1 in 1,111,111
Price: $100-$500
--]]

function match(ctx)
    local d = ctx.digits
    if not is_super_repeater(d) then
        return {matched = false}
    end

    local pair = d:sub(1, 2)
    local a = pair:sub(1, 1)
    local b = pair:sub(2, 2)

    -- Must be two different digits (Solid is CS-500, not Super Repeater)
    if a == b then return {matched = false} end

    -- Highlight alternating positions
    local pos_a = {0, 2, 4, 6}
    local pos_b = {1, 3, 5, 7}

    local highlights = {
        {positions = pos_a, color = "magenta"},
        {positions = pos_b, color = "coral"}
    }

    return {
        matched = true,
        highlights = highlights,
        connectors = {
            {from = 0, to = 2, color = "magenta", style = "arc"},
            {from = 2, to = 4, color = "magenta", style = "arc"},
            {from = 4, to = 6, color = "magenta", style = "arc"}
        },
        message = pair .. " repeated 4× (CS-Super Repeater)"
    }
end
