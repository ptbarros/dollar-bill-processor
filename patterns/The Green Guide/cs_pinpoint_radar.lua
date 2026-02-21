--[[
Pattern: CS_PINPOINT_RADAR
DisplayName: CS-Pinpoint Radar
Description: CS-Binary + CS-Radar (palindrome with two 4OAKs) where the only adjacent pair is at the center. Structure XYXYYXYX. e.g., M 41411414 M.
BookRef: CS-1320
Tier: 4
Examples: ["41411414", "10100101", "19199191"]
Odds: 1 in 450,000
Price: $10-$30
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Must use exactly 2 distinct digits
    if unique_count(d) ~= 2 then
        return {matched = false}
    end

    -- Must be a palindrome (radar)
    if not is_palindrome(d) then
        return {matched = false}
    end

    -- Each digit must appear exactly 4 times
    local counts = count_digits(d)
    for _, cnt in pairs(counts) do
        if cnt ~= 4 then
            return {matched = false}
        end
    end

    -- Center pair: positions 4 and 5 (1-indexed) must match
    if d:sub(4, 4) ~= d:sub(5, 5) then
        return {matched = false}
    end

    -- No other adjacent pairs allowed outside the center
    for i = 1, 3 do
        if d:sub(i, i) == d:sub(i + 1, i + 1) then
            return {matched = false}
        end
    end
    for i = 5, 7 do
        if d:sub(i, i) == d:sub(i + 1, i + 1) then
            return {matched = false}
        end
    end

    -- Find the two digits
    local seen = {}
    local digit_list = {}
    for i = 1, 8 do
        local ch = d:sub(i, i)
        if not seen[ch] then
            seen[ch] = true
            table.insert(digit_list, ch)
        end
    end

    local da = digit_list[1]
    local db = digit_list[2]
    local pos_a = find_digit_positions(d, da)
    local pos_b = find_digit_positions(d, db)

    return {
        matched = true,
        highlights = {
            {positions = pos_a, color = "orange"},
            {positions = pos_b, color = "coral"}
        },
        connectors = {
            {from = 0, to = 7, color = "orange", style = "arc"},
            {from = 3, to = 4, color = "coral", style = "arc"}
        },
        message = "Pinpoint radar: palindrome " .. d .. " with center pair only (CS-Pinpoint Radar)"
    }
end
