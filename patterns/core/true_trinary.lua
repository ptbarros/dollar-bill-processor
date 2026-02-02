--[[
Pattern: TRUE_TRINARY
Description: Only contains digits 0, 1, and 2
Tier: 3
Examples: ["01201210", "12021012", "00112201"]
Odds: 1 in 14,632
Price: $20-$100
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    if not only_digits(digits, "012") then
        return {matched = false}
    end

    -- Highlight each digit type
    local zeros = find_digit_positions(digits, "0")
    local ones = find_digit_positions(digits, "1")
    local twos = find_digit_positions(digits, "2")

    local highlights = {}
    if #zeros > 0 then
        table.insert(highlights, highlight(zeros, "blue", "zeros"))
    end
    if #ones > 0 then
        table.insert(highlights, highlight(ones, "cyan", "ones"))
    end
    if #twos > 0 then
        table.insert(highlights, highlight(twos, "teal", "twos"))
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = {},
        message = "True trinary (0, 1, 2 only)"
    }
end
