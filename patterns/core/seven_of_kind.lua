--[[
Pattern: SEVEN_OF_KIND
DisplayName: 7 of a Kind
Description: Seven of the same digit anywhere in the serial
Tier: 2
Examples: ["77777077", "88880888", "99909999"]
Odds: 1 in 195,122 (492 per 96M)
Price: $100-$500
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Count occurrences of each digit
    local counts = {}
    for i = 1, 8 do
        local d = digits:sub(i, i)
        counts[d] = (counts[d] or 0) + 1
    end

    -- Find digit with 7+ occurrences
    local dominant_digit = nil
    local dominant_count = 0
    for d, c in pairs(counts) do
        if c >= 7 then
            dominant_digit = d
            dominant_count = c
            break
        end
    end

    if not dominant_digit then
        return {matched = false}
    end

    -- Scattered only (Ed review): if all are contiguous it is a run (7-in-a-row or
    -- solid), which other patterns cover, so skip it here.
    local dom_positions = find_digit_positions(digits, dominant_digit)
    table.sort(dom_positions)
    if dom_positions[#dom_positions] - dom_positions[1] == #dom_positions - 1 then
        return {matched = false}
    end

    return {
        matched = true,
        -- Draw like the Nicks version: gold on the matching digits only.
        highlights = {
            highlight(dom_positions, "gold", "7 of kind")
        },
        connectors = {},
        message = string.format("%d x %s", dominant_count, dominant_digit)
    }
end
