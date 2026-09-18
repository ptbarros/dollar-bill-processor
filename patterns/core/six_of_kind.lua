--[[
Pattern: SIX_OF_KIND
Description: 6 of the same digit (anywhere)
Tier: 3
Examples: ["66666612", "11111189", "12111311"]
Odds: 1 in 17,778
Price: $20-$100
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Count occurrences of each digit
    local counts = count_digits(digits)

    -- Find digit with 6+ occurrences (but not 7+ which is seven_of_kind)
    local dominant_digit = nil
    local dominant_count = 0
    for d, c in pairs(counts) do
        if c >= 6 and c < 7 then
            dominant_digit = d
            dominant_count = c
            break
        end
    end

    if not dominant_digit then
        return {matched = false}
    end

    local dom_positions = find_digit_positions(digits, dominant_digit)
    -- Scattered only (Ed review): if all 6 are contiguous it's an 6-in-a-row, skip.
    table.sort(dom_positions)
    if dom_positions[#dom_positions] - dom_positions[1] == #dom_positions - 1 then
        return {matched = false}
    end

    return {
        matched = true,
        -- Draw like the Nicks version: gold on the matching digits only.
        highlights = {
            highlight(dom_positions, "gold", "6 of kind")
        },
        connectors = {},
        message = "6 x " .. dominant_digit
    }
end
