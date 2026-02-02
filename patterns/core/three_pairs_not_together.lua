--[[
Pattern: THREE_PAIRS_NOT_TOGETHER
Description: 3 pairs not consecutive
Tier: 4
Examples: ["33566088", "11224456"]
Odds: 1 in 200
Price: $3-$15
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Count pairs (digits appearing exactly twice)
    local counts = count_digits(digits)

    local pair_count = 0
    local pair_digits = {}
    for d, c in pairs(counts) do
        if c == 2 then
            pair_count = pair_count + 1
            table.insert(pair_digits, d)
        elseif c == 4 then
            -- 4 of a kind counts as 2 pairs
            pair_count = pair_count + 2
            table.insert(pair_digits, d)
        end
    end

    -- Need exactly 3 pairs (6 digits paired + 2 others, or various combinations)
    -- Actually for 8 digits with 3 pairs: could be 3 pairs + 1 pair = 4 pairs
    -- Or 3 distinct pairs (6 digits) + 2 singles
    -- Let's check for at least 3 pairs worth of digits

    if pair_count < 3 then
        return {matched = false}
    end

    -- Check they're not all consecutive (that would be AABBCCXX which is three_consec_pairs)
    if has_three_consecutive_pairs_start(digits) then
        return {matched = false}
    end

    -- Highlight the pairs
    local colors = {"orange", "coral", "magenta", "purple"}
    local highlights = {}
    local color_idx = 1

    for _, d in ipairs(pair_digits) do
        local positions = find_digit_positions(digits, d)
        table.insert(highlights, highlight(positions, colors[color_idx] or "gray", "pair " .. d))
        color_idx = color_idx + 1
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = {},
        message = "Three pairs (scattered)"
    }
end
