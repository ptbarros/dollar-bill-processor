--[[
Pattern: BOOKENDED_PAIR
Description: Pattern bookended by a repeating pair
Tier: 4
Examples: ["12341234", "56785678"]
Odds: 1 in 5,925
Price: $6-$20
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- This is essentially a repeater - first 4 = last 4
    if not is_repeater(digits) then
        return {matched = false}
    end

    local first_half = digits:sub(1, 4)

    return {
        matched = true,
        highlights = {
            highlight({0, 1, 2, 3}, "magenta", "first half"),
            highlight({4, 5, 6, 7}, "coral", "repeat")
        },
        group_boxes = {
            {from = 0, to = 3, color = "magenta", thickness = 2},
            {from = 4, to = 7, color = "coral", thickness = 2}
        },
        connectors = {
            connector(0, 4, "magenta", "arc"),
            connector(1, 5, "magenta", "arc"),
            connector(2, 6, "magenta", "arc"),
            connector(3, 7, "magenta", "arc")
        },
        message = "Bookended pair: " .. first_half .. " repeats"
    }
end
