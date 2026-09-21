--[[
Pattern: FOUR_CONSEC_PAIRS
DisplayName: 4 Consec Pairs
Description: Four consecutive pairs (AABBCCDD)
Tier: 3
Examples: ["11223344", "55667788", "44227733"]
Odds: 1 in 10,668 (8,999 per 96M)
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

    -- One colored box per pair (Ed review): no individual digit boxes, no lines.
    local colors = {"blue", "orange", "magenta", "red"}
    local group_boxes = {}
    for i = 0, 3 do
        local p1 = i * 2
        table.insert(group_boxes, {from = p1, to = p1 + 1, color = colors[i + 1], thickness = 3})
    end

    return {
        matched = true,
        highlights = {},
        group_boxes = group_boxes,
        connectors = {},
        message = "Four pairs: " .. a .. a .. " " .. b .. b .. " " .. c .. c .. " " .. d .. d
    }
end
