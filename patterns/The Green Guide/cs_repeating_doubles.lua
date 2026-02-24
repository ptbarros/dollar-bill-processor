--[[
Pattern: CS_REPEATING_DOUBLES
DisplayName: CS-Repeating Doubles
Description: A CS-Paired 4OAK (one digit appearing in two separate grouped pairs, i.e., AABB...AABB structure) plus two additional CS-Pairs of different digits, filling all 8 positions. Pattern is AABBAACC or similar — the 4OAK digit forms two grouped pairs separated by other pairs. e.g., M 55995533 M (55|99|55|33), M 55993355 M (55|99|33|55).
BookRef: CS-350
Tier: 4
Examples: ["55995533", "55993355", "99553355"]
Odds: 1 in 1,080
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- The 8 digits must be 4 consecutive pairs
    -- Check positions 1-2, 3-4, 5-6, 7-8 are each a pair
    local pair_digits = {}
    for i = 1, 4 do
        local a = d:sub((i-1)*2 + 1, (i-1)*2 + 1)
        local b = d:sub((i-1)*2 + 2, (i-1)*2 + 2)
        if a ~= b then
            return {matched = false}
        end
        table.insert(pair_digits, a)
    end

    -- Count how many times each digit appears in the pair positions
    local digit_pair_count = {}
    for _, digit in ipairs(pair_digits) do
        digit_pair_count[digit] = (digit_pair_count[digit] or 0) + 1
    end

    -- Must have one digit appearing in exactly 2 pairs (= 4OAK) and two digits in 1 pair each
    local doubled_pair_digit = nil
    local single_pair_digits = {}

    for digit, count in pairs(digit_pair_count) do
        if count == 2 then
            if doubled_pair_digit ~= nil then return {matched = false} end
            doubled_pair_digit = digit
        elseif count == 1 then
            table.insert(single_pair_digits, digit)
        else
            return {matched = false}
        end
    end

    if not doubled_pair_digit or #single_pair_digits ~= 2 then
        return {matched = false}
    end

    -- Build visualization
    local colors = {}
    local color_map = {[doubled_pair_digit] = "gold"}
    table.sort(single_pair_digits)
    color_map[single_pair_digits[1]] = "orange"
    color_map[single_pair_digits[2]] = "coral"

    local group_boxes = {}
    for i = 1, 4 do
        local pos_start = (i - 1) * 2
        local digit = pair_digits[i]
        table.insert(group_boxes, {
            from = pos_start,
            to = pos_start + 1,
            color = color_map[digit],
            thickness = 2
        })
    end

    local msg = doubled_pair_digit .. "x4 paired (2+2) + pairs " ..
                single_pair_digits[1] .. "," .. single_pair_digits[2] .. " (CS-Repeating Doubles)"
    return {
        matched = true,
        group_boxes = group_boxes,
        message = msg
    }
end
