--[[
Pattern: CS_BINARY
DisplayName: CS-Binary
Description: The serial contains exactly two unique digits.
BookRef: CS-910
Tier: 4
Examples: ["01010101", "11001100", "00001111"]
Price: $10-$50
--]]

function match(ctx)
    local d = ctx.digits
    local uc = unique_count(d)

    if uc ~= 2 then
        return {matched = false}
    end

    -- Find the two unique digits
    local seen = {}
    local digit_list = {}
    for i = 1, 8 do
        local ch = d:sub(i, i)
        if not seen[ch] then
            seen[ch] = true
            table.insert(digit_list, ch)
        end
    end

    local d1 = digit_list[1]
    local d2 = digit_list[2]

    -- Highlight each unique digit with a different color
    local pos1 = find_digit_positions(d, d1)
    local pos2 = find_digit_positions(d, d2)

    return {
        matched = true,
        highlights = {
            {positions = pos1, color = "blue"},
            {positions = pos2, color = "cyan"}
        },
        message = "Binary: digits " .. d1 .. " and " .. d2 .. " only (CS-Binary)"
    }
end
