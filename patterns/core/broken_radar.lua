--[[
Pattern: BROKEN_RADAR
Description: One digit away from radar
Tier: 4
Examples: ["12344322", "12345321", "15700751"]
Odds: 1 in 1,048
Price: $3-$20
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check if it's exactly one position away from being a palindrome
    local result = is_broken_palindrome(digits, 1)
    if not result then
        return {matched = false}
    end

    -- Build highlights - correct palindrome positions in orange, broken in red
    local highlights = {}
    local broken_positions = {}

    -- Flatten mismatch positions
    for _, pair in ipairs(result.positions) do
        broken_positions[pair[1]] = true
        broken_positions[pair[2]] = true
    end

    for i = 0, 7 do
        if broken_positions[i] then
            table.insert(highlights, highlight({i}, "red", "mismatch"))
        else
            table.insert(highlights, highlight({i}, "orange", "radar"))
        end
    end

    -- Add connectors showing where the palindrome breaks
    local connectors = {}
    for _, pair in ipairs(result.positions) do
        table.insert(connectors, connector(pair[1], pair[2], "red", "dashed"))
    end

    -- Add connectors for matching positions
    for i = 0, 3 do
        local j = 7 - i
        if not broken_positions[i] then
            table.insert(connectors, connector(i, j, "orange", "arc"))
        end
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = connectors,
        message = "Broken radar (1 mismatch)"
    }
end
