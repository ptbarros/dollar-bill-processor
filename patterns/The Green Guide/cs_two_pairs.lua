--[[
Pattern: CS_TWO_PAIRS
DisplayName: CS-Random Two Pairs
Description: Two different digits each occurring exactly twice in the serial, where at least one of those pairs has its digits split apart by intervening digits. Both pairs cannot be consecutive — that would qualify as CS-Two Pairs (CS-20) instead.
BookRef: CS-30
Tier: 7
Examples: ["11324526", "45016745", "91207329"]
Odds: 1 in 1,680
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    -- Find digits that appear exactly 2 times
    local paired_digits = {}
    for digit, cnt in pairs(counts) do
        if cnt == 2 then
            table.insert(paired_digits, digit)
        elseif cnt > 2 then
            -- Any digit appearing 3+ times disqualifies (would be 3OAK or higher)
            return {matched = false}
        end
    end

    -- Must have exactly 2 paired digits
    if #paired_digits ~= 2 then
        return {matched = false}
    end

    -- Build highlights; track whether at least one pair is non-adjacent
    local colors = {"orange", "coral"}
    local highlights = {}
    local connectors = {}
    local any_separated = false
    table.sort(paired_digits)
    for i, digit in ipairs(paired_digits) do
        local positions = find_digit_positions(d, digit)
        table.insert(highlights, {positions = positions, color = colors[i]})
        if positions[2] - positions[1] > 1 then
            any_separated = true
            table.insert(connectors, {from = positions[1], to = positions[2], color = colors[i], style = "arc"})
        end
    end

    -- Book requires at least one pair to be separated (not both grouped)
    if not any_separated then
        return {matched = false}
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = connectors,
        message = "Two random pairs: " .. paired_digits[1] .. paired_digits[1] .. " and " .. paired_digits[2] .. paired_digits[2] .. " (CS-30)"
    }
end
