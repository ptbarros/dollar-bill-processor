--[[
Pattern: BINARY
Description: Contains only 0s and 1s (binary number)
Tier: 4
Examples: ["10101010", "11110000", "00001111"]
Odds: 1 in 390,625 (2^8 / 10^8)
Price: $20-$100
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check that all digits are 0 or 1
    for i = 1, 8 do
        local d = digits:sub(i, i)
        if d ~= "0" and d ~= "1" then
            return {matched = false}
        end
    end

    -- Highlight 0s and 1s differently
    local highlights = {}

    for i = 0, 7 do
        local d = digits:sub(i + 1, i + 1)
        local color = d == "0" and "blue" or "cyan"
        table.insert(highlights, {
            positions = {i},
            color = color,
            label = d == "0" and "zero" or "one"
        })
    end

    -- Count 0s and 1s for message
    local zeros = 0
    local ones = 0
    for i = 1, 8 do
        if digits:sub(i, i) == "0" then
            zeros = zeros + 1
        else
            ones = ones + 1
        end
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = {},
        message = string.format("Binary: %d zeros, %d ones", zeros, ones)
    }
end
