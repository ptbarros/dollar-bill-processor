--[[
Pattern: MINI_REPEATER
Description: 3-digit pattern repeats (XX123123 format)
Tier: 4
Examples: ["94680680", "12345345", "00123123"]
Odds: 1 in 1,000
Price: $10-$30
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check pattern: XX ABC ABC where ABC repeats
    local first_group = digits:sub(3, 5)   -- positions 2-4
    local second_group = digits:sub(6, 8)  -- positions 5-7

    if first_group ~= second_group then
        return {matched = false}
    end

    -- Use group_boxes to draw a single box around each 3-digit group
    local group_boxes = {
        {from = 2, to = 4, color = "magenta", thickness = 3},
        {from = 5, to = 7, color = "magenta", thickness = 3}
    }

    return {
        matched = true,
        highlights = {},
        connectors = {},
        group_boxes = group_boxes,
        message = first_group .. " repeats"
    }
end
