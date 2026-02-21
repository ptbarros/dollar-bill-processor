--[[
Pattern: CS_SPLIT_SIX_RADAR
DisplayName: CS-Split Six Radar
Description: CS-60AK (6 of one digit) where the 2 minority digits form an adjacent pair in the interior (not at the edges). e.g., 66600666 has three 6s, pair of 0s in center, three 6s.
BookRef: CS-1300
Tier: 2
Examples: ["66600666", "33300333", "88811888"]
Price: $1,500-$10,000+
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Find the majority digit (appears exactly 6 times) and minority (exactly 2 times)
    local counts = count_digits(d)
    local majority = nil
    local minority = nil
    for digit, cnt in pairs(counts) do
        if cnt == 6 then majority = digit end
        if cnt == 2 then minority = digit end
    end

    if majority == nil or minority == nil then
        return {matched = false}
    end

    -- The 2 minority digits must be adjacent (consecutive positions)
    local min_positions = find_digit_positions(d, minority)
    if #min_positions ~= 2 then return {matched = false} end
    if min_positions[2] - min_positions[1] ~= 1 then
        return {matched = false}
    end

    -- The pair must be interior (not touching the edges: positions 1-6 in 0-indexed)
    local pair_start = min_positions[1]  -- 0-indexed
    if pair_start < 1 or pair_start > 5 then
        return {matched = false}
    end

    local maj_positions = find_digit_positions(d, majority)

    return {
        matched = true,
        highlights = {
            {positions = maj_positions, color = "gold"},
            {positions = min_positions, color = "coral"},
        },
        connectors = {
            {from = 0, to = 7, color = "gold", style = "arc"},
        },
        message = "6×" .. majority .. " split by " .. minority .. minority .. " at center (CS-Split Six Radar CS-1300)"
    }
end
