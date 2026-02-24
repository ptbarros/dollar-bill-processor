--[[
Pattern: CS_SCATTERED_LADDER
DisplayName: CS-Scattered Ladder
Description: All 8 digits are a permutation of 8 consecutive mod-10 values in scrambled (non-ladder) order. e.g., M 07634152 M or M 05734896 M.
BookRef: CS-1210
Tier: 4
Examples: ["07634152", "85412367", "57034896"]
Odds: 1 in 2,500
Price: $10-$25
--]]

function match(ctx)
    local d = ctx.digits

    -- Must have exactly 8 distinct digits (all unique)
    if unique_count(d) ~= 8 then
        return {matched = false}
    end

    -- Find k: the natural start of the 8-consecutive-mod-10 set
    local counts = {}
    for i = 1, 8 do
        counts[d:sub(i, i)] = true
    end

    local k = nil
    for candidate = 0, 9 do
        local prev = tostring((candidate - 1 + 10) % 10)
        if not counts[prev] then
            local valid = true
            for j = 0, 7 do
                if not counts[tostring((candidate + j) % 10)] then
                    valid = false
                    break
                end
            end
            if valid then
                k = candidate
                break
            end
        end
    end
    if k == nil then return {matched = false} end

    -- Build the cyclic ascending sequence for this k
    local asc_seq = ""
    for j = 0, 7 do
        asc_seq = asc_seq .. tostring((k + j) % 10)
    end

    -- Build the cyclic descending sequence for this k
    local desc_seq = ""
    for j = 0, 7 do
        desc_seq = desc_seq .. tostring((k + 7 - j + 100) % 10)
    end

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
