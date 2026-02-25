--[[
Pattern: CS_OSCILLATING_RADAR
DisplayName: CS-Oscillating Radar
Description: CS-60AK split into 3 evenly-spaced CS-Pairs separated by a CS-2OAK at positions 3 and 6. The separators must be the same digit. e.g., M 44144144 M.
BookRef: CS-1330
Tier: 2
Examples: ["44144144", "22322322", "99199199"]
Price: $1,500-$10,000+
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Structure: AABAABAAB? No, 44144144 = AA+B+AA+B+AA = 2+1+2+1+2 = 8? That's only 8 if last group is 2+1+2+1+2=8. 4,4,1,4,4,1,4,4 yes!
    -- Pattern: XX_XX_XX where _ is one different digit (same or different)
    -- Pairs at positions 0-1, 3-4, 6-7 with single separators at 2 and 5

    local a = d:sub(1, 1)  -- dominant digit

    -- Check pairs at positions 0-1 (Lua 1-2), 3-4 (Lua 4-5), 6-7 (Lua 7-8)
    if d:sub(2, 2) ~= a then return {matched = false} end
    if d:sub(4, 4) ~= a then return {matched = false} end
    if d:sub(5, 5) ~= a then return {matched = false} end
    if d:sub(7, 7) ~= a then return {matched = false} end
    if d:sub(8, 8) ~= a then return {matched = false} end

    -- Positions 3 and 6 (Lua) are the separators
    local sep1 = d:sub(3, 3)
    local sep2 = d:sub(6, 6)

    -- Separators must differ from the dominant digit and be the same (CS-2OAK)
    if sep1 == a or sep2 == a then return {matched = false} end
    if sep1 ~= sep2 then return {matched = false} end

    -- Verify count: 6 of the dominant digit
    local counts = count_digits(d)
    if (counts[a] or 0) ~= 6 then return {matched = false} end

    local pair_positions = {0, 1, 3, 4, 6, 7}
    local sep_positions = {}
    for i = 1, 8 do
        local ch = d:sub(i, i)
        if ch ~= a then
            table.insert(sep_positions, i - 1)
        end
    end

    return {
        matched = true,
        highlights = {
            {positions = pair_positions, color = "gold"},
            {positions = sep_positions, color = "coral"},
        },
        message = "3 pairs of " .. a .. "s oscillating (CS-Oscillating Radar CS-1330)"
    }
end
