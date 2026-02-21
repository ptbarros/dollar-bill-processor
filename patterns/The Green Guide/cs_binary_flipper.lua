--[[
Pattern: CS_BINARY_FLIPPER
DisplayName: CS-Binary Flipper
Description: All 8 digits are from the flip set {0,1,6,8,9} and exactly 2 distinct digits are used. e.g., M 18811181 M. Superset of CS-1040 (True Binary Flipper), which additionally requires reading the same upside-down.
BookRef: CS-1050
Tier: 3
Examples: ["18811181", "16611661", "00990099"]
Odds: 1 in 153
Price: $50-$500
--]]

function match(ctx)
    local d = ctx.digits

    -- All digits must be from the flip set: 0, 1, 6, 8, 9
    if not all_flip_valid(d) then
        return {matched = false}
    end

    -- Must use exactly 2 distinct digits
    if unique_count(d) ~= 2 then
        return {matched = false}
    end

    -- Find the two digits for visualization
    local seen = {}
    local digit_list = {}
    for i = 1, 8 do
        local ch = d:sub(i, i)
        if not seen[ch] then
            seen[ch] = true
            table.insert(digit_list, ch)
        end
    end

    local d1   = digit_list[1]
    local d2   = digit_list[2]
    local pos1 = find_digit_positions(d, d1)
    local pos2 = find_digit_positions(d, d2)

    return {
        matched = true,
        highlights = {
            {positions = pos1, color = "purple"},
            {positions = pos2, color = "cyan"}
        },
        message = "Binary flipper: only " .. d1 .. " and " .. d2 .. " (both flip-valid) (CS-Binary Flipper)"
    }
end
