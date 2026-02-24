--[[
Pattern: CS_TRI_PAIRS
DisplayName: CS-Random Tri Pairs
Description: Three different digits each appearing exactly twice in the serial. At most two of the three pairs may be adjacent — all three cannot be grouped consecutively.
BookRef: CS-50
Tier: 7
Examples: ["12031204", "12213345", "12312345"]
Odds: 1 in 2,540
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    -- Find digits that appear exactly 2 times
    local paired_digits = {}
    for digit, cnt in pairs(counts) do
        if cnt == 2 then
            table.insert(paired_digits, digit)
        elseif cnt > 2 then
            -- Any digit appearing 3+ times disqualifies
            return {matched = false}
        end
    end

    -- Must have exactly 3 paired digits (and the remaining 2 positions have unique digits)
    if #paired_digits ~= 3 then
        return {matched = false}
    end

    -- Build highlights; require at least one pair to be non-adjacent
    local colors = {"orange", "coral", "cyan"}
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

    -- Book rule: not all three pairs can be grouped
    if not any_separated then
        return {matched = false}
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = connectors,
        message = "Three random pairs (CS-50)"
    }
end
