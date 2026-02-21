--[[
Pattern: CS_TRI_PAIRS
DisplayName: CS-Random Tri Pairs
Description: Three different digits each appearing exactly twice anywhere in the FSN. At least one pair must be non-adjacent. e.g., M 12313x2xx M or M 1x232x31 M.
BookRef: CS-50
Tier: 7
Examples: ["11223345", "12213345", "12312345"]
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

    -- Build highlights
    local colors = {"orange", "coral", "cyan"}
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
        message = "Three random pairs (CS-50)"
    }
end
