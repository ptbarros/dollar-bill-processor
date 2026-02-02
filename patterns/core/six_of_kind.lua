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
            highlight(dom_positions, "gold", "6 of kind"),
            highlight(other_positions, "gray", "other")
        },
        connectors = {},
        message = "6 x " .. dominant_digit
    }
end
