--[[
Pattern: DOUBLES_BOOKEND
Description: Double digits on each end match (11XXXX11)
Tier: 4
Examples: ["11234511", "99123499", "22567822"]
Odds: 1 in 1,828
Price: $3-$15
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check first two digits are same
    if digits:sub(1, 1) ~= digits:sub(2, 2) then
        return {matched = false}
    end

    -- Check last two digits are same and match first
    if digits:sub(7, 7) ~= digits:sub(8, 8) then
        return {matched = false}
    end

    if digits:sub(1, 1) ~= digits:sub(7, 7) then
        return {matched = false}
    end

    local d = digits:sub(1, 1)

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
        message = "Doubles bookend: " .. d .. d
    }
end
