--[[
Pattern: THREE_CONSEC_PAIRS
Description: Three consecutive pairs (AABBCCXX)
Tier: 4
Examples: ["11223345", "99887712", "11883374"]
Odds: 1 in 399
Price: $3-$8
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check for AABBCC pattern at start
    if not has_three_consecutive_pairs_start(digits) then
        return {matched = false}
    end

    local a = digits:sub(1, 1)
    local b = digits:sub(3, 3)
    local c = digits:sub(5, 5)

    return {
        matched = true,
        highlights = {
            highlight({0, 1}, "orange", "pair A"),
            highlight({2, 3}, "coral", "pair B"),
            highlight({4, 5}, "magenta", "pair C")
        },
        group_boxes = {
            {from = 0, to = 1, color = "orange", thickness = 2},
            {from = 2, to = 3, color = "coral", thickness = 2},
            {from = 4, to = 5, color = "magenta", thickness = 2}
        },
        connectors = {},
        message = "Three consecutive pairs: " .. a .. a .. b .. b .. c .. c
    }
end
