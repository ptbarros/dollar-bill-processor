--[[
Pattern: CS_TRUE_BINARY
DisplayName: CS-True Binary
Description: The serial contains only 0s and 1s (exactly two digits: 0 and 1). Distinct from CS-Binary (CS-910) which allows any two unique digits.
BookRef: CS-900
Tier: 3
Examples: ["01110010", "10100101", "11001100"]
Price: $25-$500
--]]

function match(ctx)
    local d = ctx.digits

    -- Must contain only 0s and 1s
    if not only_digits(d, "01") then
        return {matched = false}
    end

    -- Must have both 0 and 1 present (not solid)
    local counts = count_digits(d)
    if not counts["0"] or not counts["1"] then
        return {matched = false}
    end

    local pos0 = find_digit_positions(d, "0")
    local pos1 = find_digit_positions(d, "1")

    return {
        matched = true,
        highlights = {
            {positions = pos0, color = "blue"},
            {positions = pos1, color = "cyan"},
        },
        message = "True Binary: only 0s and 1s (CS-900)"
    }
end
