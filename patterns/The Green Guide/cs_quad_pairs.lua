--[[
Pattern: CS_QUAD_PAIRS
DisplayName: CS-Quad Pairs (Random)
Description: All 8 digits form four pairs of different digits, with at least one pair non-adjacent. e.g., M 12432314 M or M 44123231 M. Fully-grouped AABBCCDD is a different pattern.
BookRef: CS-70
Tier: 6
Examples: ["12432314", "44123231", "12341234"]
Odds: 1 in 529
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    -- All 4 distinct digits must each appear exactly 2 times
    local paired_digits = {}
    for digit, cnt in pairs(counts) do
        if cnt ~= 2 then
            return {matched = false}
        end
        table.insert(paired_digits, digit)
    end

    -- Must have exactly 4 distinct digits (8 digits total, 2 each)
    if #paired_digits ~= 4 then
        return {matched = false}
    end

    -- Build highlights
    local colors = {"orange", "coral", "cyan", "lime"}
    local highlights = {}
    local connectors = {}
    table.sort(paired_digits)
    for i, digit in ipairs(paired_digits) do
        local positions = find_digit_positions(d, digit)
        table.insert(highlights, {positions = positions, color = colors[i]})
        -- Arc connector for separated (non-adjacent) pairs
        if positions[2] - positions[1] > 1 then
            table.insert(connectors, {from = positions[1], to = positions[2], color = colors[i], style = "arc"})
        end
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = connectors,
        message = "Four random pairs (CS-70)"
    }
end
