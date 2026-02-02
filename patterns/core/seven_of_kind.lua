--[[
Pattern: SEVEN_OF_KIND
Description: Seven of the same digit anywhere in the serial
Tier: 2
Examples: ["77777773", "18888888", "99999992"]
Odds: 1 in ~1,000,000
Price: $100-$500
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Count occurrences of each digit
    local counts = {}
    for i = 1, 8 do
        local d = digits:sub(i, i)
        counts[d] = (counts[d] or 0) + 1
    end

    -- Find digit with 7+ occurrences
    local dominant_digit = nil
    local dominant_count = 0
    for d, c in pairs(counts) do
        if c >= 7 then
            dominant_digit = d
            dominant_count = c
            break
        end
    end

    if not dominant_digit then
        return {matched = false}
    end

    -- Highlight dominant digits in gold, odd one out in red
    local highlights = {}
    local odd_positions = {}

    for i = 0, 7 do
        local d = digits:sub(i + 1, i + 1)
        if d == dominant_digit then
            table.insert(highlights, {
                positions = {i},
                color = "gold",
                label = "dominant"
            })
        else
            table.insert(highlights, {
                positions = {i},
                color = "red",
                label = "odd-out"
            })
            table.insert(odd_positions, i)
        end
    end

    -- Add connector between odd positions if there are two
    local connectors = {}
    if #odd_positions == 2 then
        table.insert(connectors, {
            from = odd_positions[1],
            to = odd_positions[2],
            color = "red",
            style = "dashed"
        })
    end

    local message = string.format("%d x %s", dominant_count, dominant_digit)
    if dominant_count == 8 then
        message = "Perfect solid - actually 8 of a kind!"
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = connectors,
        message = message
    }
end
