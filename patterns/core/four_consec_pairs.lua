--[[
Pattern: FOUR_CONSEC_PAIRS
Description: Four consecutive pairs (AABBCCDD)
Tier: 3
Examples: ["11223344", "55667788", "44227733"]
Odds: 1 in 21,164
Price: $20-$100
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check for AABBCCDD pattern
    if not has_four_consecutive_pairs(digits) then
        return {matched = false}
    end

    local a = digits:sub(1, 1)
    local b = digits:sub(3, 3)
    local c = digits:sub(5, 5)
    local d = digits:sub(7, 7)

    return {
        matched = true,
        highlights = {
            highlight({0, 1}, "orange", "pair A"),
            highlight({2, 3}, "coral", "pair B"),
            highlight({4, 5}, "magenta", "pair C"),
            highlight({6, 7}, "purple", "pair D")
        },
        group_boxes = {
            {from = 0, to = 1, color = "orange", thickness = 2},
            {from = 2, to = 3, color = "coral", thickness = 2},
            {from = 4, to = 5, color = "magenta", thickness = 2},
            {from = 6, to = 7, color = "purple", thickness = 2}
        },
        connectors = {},
        message = "Four pairs: " .. a .. a .. " " .. b .. b .. " " .. c .. c .. " " .. d .. d
    }
end
