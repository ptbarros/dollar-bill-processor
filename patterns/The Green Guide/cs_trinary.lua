--[[
Pattern: CS_TRINARY
DisplayName: CS-Trinary
Description: The serial contains exactly three unique digits.
BookRef: CS-940
Tier: 5
Examples: ["01201201", "11220011", "00011122"]
Price: $5-$25
--]]

function match(ctx)
    local d = ctx.digits
    local uc = unique_count(d)

    if uc ~= 3 then
        return {matched = false}
    end

    -- Find the three unique digits
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
    local d3 = digit_list[3]

    local pos1 = find_digit_positions(d, d1)
    local pos2 = find_digit_positions(d, d2)
    local pos3 = find_digit_positions(d, d3)

    local tri_colors = {"cyan", "orange", "lime"}
    return {
        matched = true,
        highlights = {
            {positions = pos1, color = tri_colors[1]},
            {positions = pos2, color = tri_colors[2]},
            {positions = pos3, color = tri_colors[3]}
        },
        message = "Trinary: digits " .. d1 .. ", " .. d2 .. ", " .. d3 .. " only (CS-Trinary)"
    }
end
