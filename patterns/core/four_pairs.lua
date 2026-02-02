--[[
Pattern: FOUR_PAIRS
Description: Four pairs of digits (AABBCCDD pattern, not necessarily consecutive)
Tier: 4
Examples: ["11223344", "99887766", "31898391"]
Odds: 1 in 188
Price: $3-$8
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local counts = count_digits(digits)

    -- Count how many digits appear exactly twice
    local pairs_count = 0
    local pair_digits = {}
    for d, c in pairs(counts) do
        if c == 2 then
            pairs_count = pairs_count + 1
            table.insert(pair_digits, d)
        elseif c == 4 then
            -- Count as two pairs
            pairs_count = pairs_count + 2
            table.insert(pair_digits, d)
            table.insert(pair_digits, d)
        end
    end

    if pairs_count ~= 4 then
        return {matched = false}
    end

    -- Color each unique digit's positions
    local colors = {"orange", "coral", "magenta", "purple"}
    local highlights = {}
    local seen = {}
    local color_idx = 1

    for i = 0, 7 do
        local d = digits:sub(i + 1, i + 1)
        if not seen[d] then
            seen[d] = colors[color_idx]
            color_idx = color_idx + 1
        end
    end

    for d, color in pairs(seen) do
        local pos = find_digit_positions(digits, d)
        table.insert(highlights, highlight(pos, color, "pair " .. d))
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = {},
        message = "Four pairs"
    }
end
