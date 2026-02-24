--[[
Pattern: CS_TRI_PAIRS_GROUPED
DisplayName: CS-Tri Pairs
Description: Three different digits each appearing exactly twice, where all three pairs are internally grouped (adjacent). The three pairs do not need to be adjacent to each other. e.g., M 112233xx M or M x22x3311 M. Distinguished from CS-50 (Random Tri Pairs) where at least one pair is split.
BookRef: CS-40
Tier: 7
Examples: ["11224500", "11992276", "66778900"]
Odds: 1 in 15,120
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    -- Find digits appearing exactly 2 times
    local paired_digits = {}
    for digit, cnt in pairs(counts) do
        if cnt == 2 then
            table.insert(paired_digits, digit)
        elseif cnt > 2 then
            return {matched = false}
        end
    end

    -- Must have exactly 3 paired digits
    if #paired_digits ~= 3 then
        return {matched = false}
    end

    -- All three pairs must be internally adjacent (grouped)
    table.sort(paired_digits)
    local colors = {"orange", "coral", "cyan"}
    local highlights = {}
    local group_boxes = {}

    for i, digit in ipairs(paired_digits) do
        local positions = find_digit_positions(d, digit)
        -- Both positions must be adjacent
        if positions[2] - positions[1] ~= 1 then
            return {matched = false}
        end
        table.insert(highlights, {positions = positions, color = colors[i]})
        table.insert(group_boxes, {from = positions[1], to = positions[2], color = colors[i], thickness = 2})
    end

    local msg = "Three grouped pairs: " ..
        paired_digits[1] .. paired_digits[1] .. ", " ..
        paired_digits[2] .. paired_digits[2] .. ", " ..
        paired_digits[3] .. paired_digits[3] .. " (CS-Tri Pairs)"
    return {
        matched = true,
        highlights = highlights,
        group_boxes = group_boxes,
        message = msg
    }
end
