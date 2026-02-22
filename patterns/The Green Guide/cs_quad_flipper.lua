--[[
Pattern: CS_QUAD_FLIPPER
DisplayName: CS-Quad Flipper
Description: All 8 digits are from the flip set {0,1,6,8,9} and exactly 4 distinct digits are used. e.g., M 01899901 M.
BookRef: CS-1070
Tier: 7
Examples: ["01899901", "01689610", "18906690"]
Odds: 1 in 7
Price: $0-$5
--]]

function match(ctx)
    local d = ctx.digits

    -- All digits must be from the flip set: 0, 1, 6, 8, 9
    if not all_flip_valid(d) then
        return {matched = false}
    end

    -- Must use exactly 4 distinct digits
    if unique_count(d) ~= 4 then
        return {matched = false}
    end

    -- Find the four digits for visualization
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
    local d4 = digit_list[4]

    local pos1 = find_digit_positions(d, d1)
    local pos2 = find_digit_positions(d, d2)
    local pos3 = find_digit_positions(d, d3)
    local pos4 = find_digit_positions(d, d4)

    return {
        matched = true,
        highlights = {
            {positions = pos1, color = "purple"},
            {positions = pos2, color = "teal"},
            {positions = pos3, color = "cyan"},
            {positions = pos4, color = "magenta"}
        },
        message = "Quad flipper: digits " .. d1 .. ", " .. d2 .. ", " .. d3 .. ", " .. d4 .. " (all flip-valid) (CS-Quad Flipper)"
    }
end
