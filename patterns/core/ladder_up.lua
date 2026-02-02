--[[
Pattern: LADDER_UP
Description: Perfect ascending sequence (each digit is +1 from previous)
Tier: 2
Examples: ["01234567", "12345678", "23456789"]
Odds: 1 in 100,000,000 (only 3 possible)
Price: $500-$2,000+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check ascending sequence
    for i = 1, 7 do
        local curr = tonumber(digits:sub(i, i))
        local next_d = tonumber(digits:sub(i + 1, i + 1))
        if next_d ~= curr + 1 then
            return {matched = false}
        end
    end

    -- Highlight all digits in lime showing the sequence
    local highlights = {}
    local connectors = {}

    for i = 0, 7 do
        table.insert(highlights, {
            positions = {i},
            color = "lime",
            label = "step"
        })
    end

    -- Add flow connectors between consecutive positions
    for i = 0, 6 do
        table.insert(connectors, {
            from = i,
            to = i + 1,
            color = "lime",
            style = "arrow"
        })
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = connectors,
        message = "Perfect ascending ladder"
    }
end
