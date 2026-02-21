--[[
Pattern: CS_TRINARY_FLIPPER
DisplayName: CS-Trinary Flipper
Description: All 8 digits are from the flip set {0,1,6,8,9} and exactly 3 distinct digits are used. e.g., M 01810081 M.
Tier: 7
Examples: ["01810081", "08880000", "61916191"]
Odds: 1 in 21
Price: $0-$5
--]]

function match(ctx)
    local d = ctx.digits

    -- All digits must be from the flip set: 0, 1, 6, 8, 9
    if not all_flip_valid(d) then
        return {matched = false}
    end

    -- Must use exactly 3 distinct digits
    if unique_count(d) ~= 3 then
        return {matched = false}
    end

    -- Find the three digits for visualization
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

    return {
        matched = true,
        highlights = {
            {positions = pos1, color = "purple"},
            {positions = pos2, color = "teal"},
            {positions = pos3, color = "cyan"}
        },
        message = "Trinary flipper: digits " .. d1 .. ", " .. d2 .. ", " .. d3 .. " (all flip-valid) (CS-Trinary Flipper)"
    }
end
