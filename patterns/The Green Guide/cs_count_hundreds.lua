--[[
Pattern: CS_COUNT_HUNDREDS
DisplayName: CS-Count Hundreds
Description: Three groups (3+3+2 digits) where the leading digit of each group counts up or down by 1. The base digits (positions 2-3 of each group) stay the same. e.g., M 411 511 61 M or M 123 223 32 M.
BookRef: CS-830
Tier: 3
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
        message = "Hundreds count " .. direction .. ": " .. c1 .. "xx " .. c2 .. "xx " .. c3 .. "x (CS-830)"
    }
end
