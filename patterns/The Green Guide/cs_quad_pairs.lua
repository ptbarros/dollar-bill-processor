--[[
Pattern: CS_QUAD_PAIRS
DisplayName: CS-Random Quad Pairs
Description: Four pairs of different digits spanning all 8 positions, with at least one pair having its two digits non-adjacent. A fully-grouped AABBCCDD arrangement qualifies as CS-Quad Pairs (CS-60) instead.
BookRef: CS-70
Tier: 6
Examples: ["13241324", "23413241", "12341234"]
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

    -- Build highlights; require at least one pair to be non-adjacent (excludes AABBCCDD → CS-60)
    local colors = {"orange", "coral", "cyan", "lime"}
    local highlights = {}
    local connectors = {}
    local any_separated = false
    table.sort(paired_digits)
    for i, digit in ipairs(paired_digits) do
        local positions = find_digit_positions(d, digit)
        table.insert(highlights, {positions = positions, color = colors[i]})
        if positions[2] - positions[1] > 1 then
            any_separated = true
            table.insert(connectors, {from = positions[1], to = positions[2], color = colors[i], style = "arc"})
        end
    end

    -- Book rule: at least one pair must be separated (fully-grouped is CS-60)
    if not any_separated then
        return {matched = false}
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = connectors,
        message = "Four random pairs (CS-70)"
    }
end
