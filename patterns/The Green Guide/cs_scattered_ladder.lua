--[[
Pattern: CS_SCATTERED_LADDER
DisplayName: CS-Scattered Ladder
Description: All 8 digits are a permutation of a straight run of 8 consecutive values (0-7, 1-8, or 2-9) in scrambled (non-ladder) order. No 9->0 wrap (that is a Looping Ladder). e.g., M 07634152 M or M 25734896 M.
BookRef: CS-1210
Tier: 4
Examples: ["07634152", "85412367", "57234896"]
Odds: 1 in 2,500
Price: $10-$25
--]]

function match(ctx)
    local d = ctx.digits

    -- Must have exactly 8 distinct digits (all unique)
    if unique_count(d) ~= 8 then
        return {matched = false}
    end

    -- Find the min and max digit
    local lo, hi = 10, -1
    for i = 1, 8 do
        local n = tonumber(d:sub(i, i))
        if n < lo then lo = n end
        if n > hi then hi = n end
    end

    -- Require a STRAIGHT run of 8 consecutive values (no 9->0 wrap).
    -- With 8 distinct digits, hi - lo == 7 guarantees the set is exactly {lo..hi},
    -- i.e. one of 0-7, 1-8, 2-9. Looping (wrapping) runs are CS-Looping Ladder.
    if hi - lo ~= 7 then
        return {matched = false}
    end
    local k = lo

    -- Build the ascending straight sequence for this run
    local asc_seq = ""
    for j = 0, 7 do
        asc_seq = asc_seq .. tostring(k + j)
    end

    -- Build the descending straight sequence for this run
    local desc_seq = string.reverse(asc_seq)

    -- Exclude any cyclic ascending rotation (CS-Ascending Ladder or CS-Ascending Looping Ladder)
    if string.find(asc_seq .. asc_seq, d, 1, true) then
        return {matched = false}
    end

    -- Exclude any cyclic descending rotation (CS-Descending Ladder or CS-Descending Looping Ladder)
    if string.find(desc_seq .. desc_seq, d, 1, true) then
        return {matched = false}
    end

    -- Build highlights: gradient from lime (low) to green (high) by digit value
    local gradient_colors = {"lime", "lime", "lime", "lime", "green", "green", "green", "green"}
    local highlights = {}
    for j = 0, 7 do
        local n = (k + j) % 10
        local pos = {}
        for i = 1, 8 do
            if tonumber(d:sub(i, i)) == n then
                table.insert(pos, i - 1)
            end
        end
        local color = gradient_colors[j + 1]
        table.insert(highlights, {positions = pos, color = color})
    end

    local max_digit = (k + 7) % 10

    return {
        matched = true,
        highlights = highlights,
        message = "Scattered ladder: digits k=" .. k .. " set, scrambled order (CS-Scattered Ladder)"
    }
end
