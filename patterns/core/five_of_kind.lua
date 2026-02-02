--[[
Pattern: FIVE_OF_KIND
Description: 5 of the same digit (anywhere)
Tier: 4
Examples: ["55555123", "49446144", "12555553"]
Odds: 1 in 232
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

    -- Highlight dominant digits in gold, others in gray
    local dom_positions = find_digit_positions(digits, dominant_digit)
    local other_positions = {}
    for i = 0, 7 do
        local found = false
        for _, p in ipairs(dom_positions) do
            if p == i then found = true; break end
        end
        if not found then
            table.insert(other_positions, i)
        end
    end

    return {
        matched = true,
        highlights = {
            highlight(dom_positions, "gold", "5 of kind"),
            highlight(other_positions, "gray", "other")
        },
        connectors = {},
        message = "5 x " .. dominant_digit
    }
end
