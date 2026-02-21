--[[
Pattern: CS_FULL_RADAR
DisplayName: CS-Full Radar
Description: All 8 positions form a palindrome — reads the same forwards and backwards.
BookRef: CS-1270
Tier: 3
Examples: ["12344321", "56477465", "10011001"]
Odds: 1 in 10,000
Price: $25-$150
--]]

function match(ctx)
    local d = ctx.digits
    if not is_palindrome(d) then
        return {matched = false}
    end

    -- Highlight mirrored pairs
    local highlights = {}
    local pair_colors = {"orange", "coral", "cyan", "lime"}
    for i = 0, 3 do
        local mirror = 7 - i
        local color = pair_colors[(i % #pair_colors) + 1]
        if i == mirror then
            table.insert(highlights, {positions = {i}, color = color})
        else
            table.insert(highlights, {positions = {i, mirror}, color = color})
        end
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = {
            {from = 0, to = 7, color = "purple", style = "arc"},
            {from = 1, to = 6, color = "purple", style = "arc"},
            {from = 2, to = 5, color = "purple", style = "arc"},
            {from = 3, to = 4, color = "purple", style = "arc"}
        },
        message = "Full 8-digit palindrome (CS-Full Radar)"
    }
end
