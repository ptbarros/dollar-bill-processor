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

    -- Draw like Double Double (Ed review): a bracketed colored box per pair.
    local colors = {"teal", "cyan", "blue", "purple"}
    local highlights = {}
    local connectors = {}
    for i = 0, 3 do
        local p1, p2 = i * 2, i * 2 + 1
        table.insert(highlights, {positions = {p1, p2}, color = colors[i + 1], label = "pair"})
        table.insert(connectors, {from = p1, to = p2, color = colors[i + 1], style = "bracket"})
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = connectors,
        message = "Four pairs: " .. a .. a .. " " .. b .. b .. " " .. c .. c .. " " .. d .. d
    }
end
