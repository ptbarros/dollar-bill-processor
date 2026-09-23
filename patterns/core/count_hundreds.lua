--[[
Pattern: COUNT_HUNDREDS
DisplayName: Count by 100
Description: The serial breaks into groups whose leading digit steps up or down by one while the rest of each group stays the same (e.g. 411·511·61).
Tier: 4
Odds: 1 in 61,538 (1,560 per 96M)
Examples: ["12322332", "32322312", "41151161"]
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Groups of 3+3+2: positions 0-2, 3-5, 6-7
    -- Counting digit: positions 0, 3, 6
    -- Base digits: positions 1-2 must equal 4-5, and position 7 must equal position 1
    local c1 = tonumber(d:sub(1, 1))
    local c2 = tonumber(d:sub(4, 4))
    local c3 = tonumber(d:sub(7, 7))

    -- Base check: positions 1-2 == 4-5, position 7 == position 1
    if d:sub(2, 3) ~= d:sub(5, 6) then return {matched = false} end
    if d:sub(8, 8) ~= d:sub(2, 2) then return {matched = false} end

    -- Counting digits must form ±1 sequence
    local step = c2 - c1
    if step ~= 1 and step ~= -1 then return {matched = false} end
    if c3 - c2 ~= step then return {matched = false} end

    local direction = step == 1 and "up" or "down"
    return {
        matched = true,
        highlights = {
            {positions = {0, 3, 6}, color = "gold"}
        },
        group_boxes = {
            {from = 0, to = 2, color = "cyan", thickness = 2},
            {from = 3, to = 5, color = "cyan", thickness = 2},
            {from = 6, to = 7, color = "coral", thickness = 2},
        },
        connectors = {
            {from = 0, to = 3, color = "lime", style = "arc"},
            {from = 3, to = 6, color = "lime", style = "arc"},
        },
        message = "Hundreds count " .. direction .. ": " .. c1 .. "xx " .. c2 .. "xx " .. c3 .. "x"
    }
end
