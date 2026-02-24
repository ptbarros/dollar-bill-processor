--[[
Pattern: CS_QUINARY_FLIPPER
DisplayName: CS-Quinary Flipper
Description: All 8 digits are from the flip set {0,1,6,8,9} AND all five flip digits are present. The serial can be read upside down. e.g., M 01698900 M.
BookRef: CS-1080
Tier: 6
Examples: ["01698900", "91806100", "61890001"]
Odds: 1 in 126,000
Price: $0-$5
--]]

function match(ctx)
    local d = ctx.digits

    -- All digits must be from the flip set {0,1,6,8,9}
    if not all_flip_valid(d) then return {matched = false} end

    -- Must use exactly 5 distinct digits (all five flip digits present)
    if unique_count(d) ~= 5 then return {matched = false} end

    -- Build visualization with each flip digit in a distinct color
    local flip_colors = {["0"] = "blue", ["1"] = "cyan", ["6"] = "orange", ["8"] = "gold", ["9"] = "magenta"}
    local highlights = {}
    for digit, color in pairs(flip_colors) do
        local positions = find_digit_positions(d, digit)
        if #positions > 0 then
            table.insert(highlights, {positions = positions, color = color})
        end
    end

    return {
        matched = true,
        highlights = highlights,
        message = "All 5 flip digits present: 0,1,6,8,9 (CS-Quinary Flipper)"
    }
end
