--[[
Pattern: TWIN_PAIR_DOUBLES
Description: Twin pair doubles (4141, 9696, 1212, etc.)
Tier: 4
Examples: ["41419696", "12121919"]
Odds: 1 in 14,814
Price: $5-$30
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Pattern: ABAB CDCD where AB and CD are each a pair that repeats
    -- Check first 4 digits form ABAB pattern
    local a1 = digits:sub(1, 1)
    local b1 = digits:sub(2, 2)
    if digits:sub(3, 3) ~= a1 or digits:sub(4, 4) ~= b1 then
        return {matched = false}
    end

    -- Check last 4 digits form CDCD pattern
    local c = digits:sub(5, 5)
    local d = digits:sub(6, 6)
    if digits:sub(7, 7) ~= c or digits:sub(8, 8) ~= d then
        return {matched = false}
    end

    -- A must equal B for "twin pair" (so first half is AAAA or ABAB with distinct A,B)
    -- Actually, pattern is ABAB CDCD - two different alternating pairs

    return {
        matched = true,
        highlights = {
            highlight({0, 2}, "orange", "A"),
            highlight({1, 3}, "coral", "B"),
            highlight({4, 6}, "magenta", "C"),
            highlight({5, 7}, "purple", "D")
        },
        group_boxes = {
            {from = 0, to = 3, color = "orange", thickness = 2},
            {from = 4, to = 7, color = "magenta", thickness = 2}
        },
        connectors = {
            connector(0, 2, "orange", "line"),
            connector(1, 3, "coral", "line"),
            connector(4, 6, "magenta", "line"),
            connector(5, 7, "purple", "line")
        },
        message = "Twin pair doubles: " .. a1 .. b1 .. a1 .. b1 .. " + " .. c .. d .. c .. d
    }
end
