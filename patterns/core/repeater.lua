--[[
Pattern: REPEATER
Description: First 4 digits repeat exactly (ABCDABCD)
Tier: 3
Examples: ["12341234", "56785678", "90129012"]
Odds: 1 in 10,000
Price: $50-$200
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check if first half equals second half
    local first_half = digits:sub(1, 4)
    local second_half = digits:sub(5, 8)

    if first_half ~= second_half then
        return {matched = false}
    end

    -- Highlight all digits in magenta
    local highlights = {}
    local connectors = {}

    -- First group
    table.insert(highlights, {
        positions = {0, 1, 2, 3},
        color = "magenta",
        label = "first"
    })

    -- Second group
    table.insert(highlights, {
        positions = {4, 5, 6, 7},
        color = "magenta",
        label = "repeat"
    })

    -- Add connectors showing the repetition
    for i = 0, 3 do
        table.insert(connectors, {
            from = i,
            to = i + 4,
            color = "magenta",
            style = "line"
        })
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = connectors,
        message = first_half .. " repeats"
    }
end
