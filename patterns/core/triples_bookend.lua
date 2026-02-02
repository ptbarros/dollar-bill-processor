--[[
Pattern: TRIPLES_BOOKEND
Description: Triple on each end (111XX111)
Tier: 3
Examples: ["11122111", "99912999", "33345333"]
Odds: 1 in 148,148
Price: $20-$150
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check first three digits are same
    local d = digits:sub(1, 1)
    if digits:sub(2, 2) ~= d or digits:sub(3, 3) ~= d then
        return {matched = false}
    end

    -- Check last three digits are same and match first
    if digits:sub(6, 6) ~= d or digits:sub(7, 7) ~= d or digits:sub(8, 8) ~= d then
        return {matched = false}
    end

    return {
        matched = true,
        highlights = {},
        connectors = {
            connector(1, 6, "gold", "arc")
        },
        group_boxes = {
            {from = 0, to = 2, color = "gold", thickness = 2},
            {from = 5, to = 7, color = "gold", thickness = 2}
        },
        message = "Triples bookend: " .. d .. d .. d
    }
end
