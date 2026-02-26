--[[
Pattern: CS_SHOTGUN_RADAR
DisplayName: CS-Shotgun Radar
Description: A radar with exactly 3 of 4 mirror pairs matching. The non-matching pair must have different digits (not a CS-Pair). The 3 matching pairs must use at least 2 distinct digit values.
BookRef: CS-1340
Tier: 2
Examples: ["12301321", "56701765", "80101108"]
Odds: 1 in 5,000
Price: $1,500-$10,000
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Check all 4 mirror pairs (0<>7, 1<>6, 2<>5, 3<>4)
    local matching_pairs = {}
    local non_matching_pairs = {}
    for i = 0, 3 do
        local left = d:sub(i + 1, i + 1)
        local right = d:sub(8 - i, 8 - i)
        if left == right then
            table.insert(matching_pairs, {left_pos = i, right_pos = 7 - i, digit = left})
        else
            table.insert(non_matching_pairs, {left_pos = i, right_pos = 7 - i, left_digit = left, right_digit = right})
        end
    end

    -- Exactly 3 of 4 mirror pairs must match
    if #matching_pairs ~= 3 then return {matched = false} end

    -- The non-matching pair must have different digits (cannot be a CS-Pair)
    local nm = non_matching_pairs[1]
    if nm.left_digit == nm.right_digit then return {matched = false} end

    -- The 3 matching pairs must use at least 2 distinct digit values
    local pair_digits = {}
    for _, p in ipairs(matching_pairs) do
        pair_digits[p.digit] = true
    end
    local distinct = 0
    for _ in pairs(pair_digits) do distinct = distinct + 1 end
    if distinct < 2 then return {matched = false} end

    -- Build visualization
    local highlights = {}
    local connectors = {}
    local pair_colors = {"orange", "coral", "cyan"}
    for idx, p in ipairs(matching_pairs) do
        local color = pair_colors[idx]
        table.insert(highlights, {positions = {p.left_pos, p.right_pos}, color = color})
        table.insert(connectors, {from = p.left_pos, to = p.right_pos, color = color, style = "arc"})
    end
    -- Gray for the non-matching pair
    table.insert(highlights, {positions = {nm.left_pos, nm.right_pos}, color = "gray"})

    return {
        matched = true,
        highlights = highlights,
        connectors = connectors,
        message = "3 of 4 mirror pairs match (Shotgun Radar)"
    }
end
