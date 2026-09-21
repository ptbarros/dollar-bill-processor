--[[
Pattern: CS_QUINARY_FLIPPER
DisplayName: CS-Quinary Flipper
Description: Uses all five of the digits that still look like digits upside-down — 0, 1, 6, 8 and 9 — and nothing else, so the whole note can be read upside-down (e.g. 0169·8900).
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
    -- Emit highlights in reading order (first appearance), not pairs(flip_colors) order,
    -- so the overlay's first-seen colour remap is deterministic per launch.
    local highlights = {}
    local seen = {}
    for i = 1, 8 do
        local digit = d:sub(i, i)
        if not seen[digit] then
            seen[digit] = true
            table.insert(highlights, {positions = find_digit_positions(d, digit), color = flip_colors[digit]})
        end
    end

    return {
        matched = true,
        highlights = highlights,
        message = "All 5 flip digits present: 0,1,6,8,9 (CS-Quinary Flipper)"
    }
end
