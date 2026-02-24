--[[
Pattern: CS_GROUPED_TWO_PAIRS
DisplayName: CS-Two Pairs
Description: Two CS-Pairs (each pair grouped consecutively) of different digits anywhere in the serial. Both pairs must be adjacent (grouped). e.g., M xx2299xx M or M 22xxxx99 M. Distinguished from CS-30 (Random Two Pairs) where at least one pair is split.
BookRef: CS-20
Tier: 7
Examples: ["11223456", "12341122", "99123456"]
Odds: 1 in 2,520
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    -- Find digits appearing exactly 2 times, where both are adjacent (grouped pair)
    local grouped_pairs = {}
    for digit, cnt in pairs(counts) do
        if cnt == 2 then
            local positions = find_digit_positions(d, digit)
            if positions[2] - positions[1] == 1 then
                -- Adjacent = grouped pair
                table.insert(grouped_pairs, {digit = digit, pos1 = positions[1], pos2 = positions[2]})
            end
        elseif cnt > 2 then
            return {matched = false}
        end
    end

    -- Must have exactly 2 grouped pairs (different digits)
    if #grouped_pairs ~= 2 then
        return {matched = false}
    end

    table.sort(grouped_pairs, function(a, b) return a.digit < b.digit end)

    local colors = {"orange", "coral"}
    local highlights = {}
    local group_boxes = {}

    for i, pair in ipairs(grouped_pairs) do
        table.insert(highlights, {positions = {pair.pos1, pair.pos2}, color = colors[i]})
        table.insert(group_boxes, {from = pair.pos1, to = pair.pos2, color = colors[i], thickness = 2})
    end

    local msg = "Two grouped pairs: " .. grouped_pairs[1].digit .. grouped_pairs[1].digit ..
                " and " .. grouped_pairs[2].digit .. grouped_pairs[2].digit .. " (CS-Two Pairs)"
    return {
        matched = true,
        highlights = highlights,
        group_boxes = group_boxes,
        message = msg
    }
end
