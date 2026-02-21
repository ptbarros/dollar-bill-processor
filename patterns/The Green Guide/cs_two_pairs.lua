--[[
Pattern: CS_TWO_PAIRS
DisplayName: CS-Two Pairs (Random)
Description: Two different digits each appearing exactly twice, with at least one pair non-adjacent. e.g., M xx229x9x M or M 2x9x2xx9 M.
BookRef: CS-30
Tier: 7
Examples: ["11234526", "12134256", "91234529"]
Odds: 1 in 1,680
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
            -- Any digit appearing 3+ times disqualifies (would be 3OAK or higher)
            return {matched = false}
        end
    end

    -- Must have exactly 2 paired digits
    if #paired_digits ~= 2 then
        return {matched = false}
    end

    -- Build highlights
    local colors = {"orange", "coral"}
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
        message = "Two random pairs: " .. paired_digits[1] .. paired_digits[1] .. " and " .. paired_digits[2] .. paired_digits[2] .. " (CS-30)"
    }
end
