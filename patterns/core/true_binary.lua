--[[
Pattern: TRUE_BINARY
Description: Binary using only 0 and 1
Tier: 3
Examples: ["01101001", "10010110", "10100100"]
Odds: 1 in 375,000
Price: $50-$4,500
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    if not only_digits(digits, "01") then
        return {matched = false}
    end

    -- Highlight 0s and 1s in different colors
    local zeros = find_digit_positions(digits, "0")
    local ones = find_digit_positions(digits, "1")

    return {
        matched = true,
        highlights = {
            highlight(zeros, "blue", "zeros"),
            highlight(ones, "cyan", "ones")
        },
        connectors = {},
        message = "True binary (0s and 1s only)"
    }
end
