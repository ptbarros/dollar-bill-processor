--[[
Pattern: RADAR
Description: Palindrome - reads same forwards and backwards
Tier: 3
Examples: ["12344321", "01233210", "45677654"]
Odds: 1 in 10,000
Price: $50-$300
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check palindrome
    local rev = string.reverse(digits)
    if digits ~= rev then
        return {matched = false}
    end

    -- Build highlights and connectors for paired positions
    local colors = {"orange", "coral", "gold", "salmon"}
    local highlights = {}
    local connectors = {}

    for i = 0, 3 do
        local j = 7 - i
        local pair_color = colors[i + 1]

        -- Highlight both positions in the pair
        table.insert(highlights, {
            positions = {i, j},
            color = pair_color,
            label = "pair"
        })

        -- Add connector arc between the pair
        table.insert(connectors, {
            from = i,
            to = j,
            color = pair_color,
            style = "arc"
        })
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = connectors,
        message = "Perfect palindrome"
    }
end
