--[[
Pattern: DOUBLE_DOUBLE
Description: Four consecutive pairs (AABBCCDD format)
Tier: 4
Examples: ["11223344", "00112233", "55667788"]
Odds: 1 in 10,000
Price: $20-$100
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check for four consecutive pairs
    for i = 0, 3 do
        local pos = i * 2 + 1
        local d1 = digits:sub(pos, pos)
        local d2 = digits:sub(pos + 1, pos + 1)
        if d1 ~= d2 then
            return {matched = false}
        end
    end

    -- Highlight each pair with different colors
    local colors = {"teal", "cyan", "blue", "purple"}
    local highlights = {}
    local connectors = {}

    for i = 0, 3 do
        local pos1 = i * 2
        local pos2 = i * 2 + 1
        local color = colors[i + 1]

        table.insert(highlights, {
            positions = {pos1, pos2},
            color = color,
            label = "pair"
        })

        -- Connect the pair
        table.insert(connectors, {
            from = pos1,
            to = pos2,
            color = color,
            style = "bracket"
        })
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = connectors,
        message = "Four consecutive pairs"
    }
end
