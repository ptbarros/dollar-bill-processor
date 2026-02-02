--[[
Pattern: DOUBLE_QUADS
Description: Two groups of 4 identical digits (AAAABBBB)
Tier: 2
Examples: ["11112222", "99990000", "22226666"]
Odds: 1 in 1,185,185
Price: $80-$1,500
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check first 4 are same
    local a = digits:sub(1, 1)
    for i = 2, 4 do
        if digits:sub(i, i) ~= a then
            return {matched = false}
        end
    end

    -- Check last 4 are same
    local b = digits:sub(5, 5)
    for i = 6, 8 do
        if digits:sub(i, i) ~= b then
            return {matched = false}
        end
    end

    -- They must be different (otherwise it's a solid)
    if a == b then
        return {matched = false}
    end

    return {
        matched = true,
        highlights = {
            highlight({0, 1, 2, 3}, "gold", "first quad"),
            highlight({4, 5, 6, 7}, "coral", "second quad")
        },
        group_boxes = {
            {from = 0, to = 3, color = "gold", thickness = 3},
            {from = 4, to = 7, color = "coral", thickness = 3}
        },
        connectors = {},
        message = "Double quads: " .. a .. a .. a .. a .. " + " .. b .. b .. b .. b
    }
end
