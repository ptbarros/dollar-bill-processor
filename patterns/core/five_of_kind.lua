--[[
Pattern: FIVE_OF_KIND
DisplayName: 5 of a Kind
Description: 5 of the same digit (anywhere)
Tier: 8
Examples: ["50505055", "49446144", "50550505"]
Odds: 1 in 269 (356,994 per 96M)
Price: $3-$25
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Count occurrences of each digit
    local counts = count_digits(digits)

    -- Find digit with exactly 5 occurrences
    local dominant_digit = nil
    for d, c in pairs(counts) do
        if c == 5 then
            dominant_digit = d
            break
        end
    end

    if not dominant_digit then
        return {matched = false}
    end

    local dom_positions = find_digit_positions(digits, dominant_digit)
    -- Scattered only (Ed review): if all 5 are contiguous it's an 5-in-a-row, skip.
    table.sort(dom_positions)
    if dom_positions[#dom_positions] - dom_positions[1] == #dom_positions - 1 then
        return {matched = false}
    end

    return {
        matched = true,
        -- Draw like the Nicks version: gold on the matching digits only.
        highlights = {
            highlight(dom_positions, "gold", "5 of kind")
        },
        connectors = {},
        message = "5 x " .. dominant_digit
    }
end
