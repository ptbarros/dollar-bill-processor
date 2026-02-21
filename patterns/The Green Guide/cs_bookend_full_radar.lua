--[[
Pattern: CS_BOOKEND_FULL_RADAR
DisplayName: CS-Bookend Full Radar
Description: A CS-Full Radar (palindrome) with a CS-40AK (two CS-Pairs) at each end: e.g., M 44133144 M. The outer 2 digits at each end match (forming a pair bookend), and the serial is a palindrome.
BookRef: CS-1280
Tier: 4
Examples: ["44133144", "22155122", "88100188"]
Price: $5-$50
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Must be a palindrome (CS-Full Radar)
    if not is_palindrome(d) then
        return {matched = false}
    end

    -- Outer pair: positions 0-1 must be the same digit
    local outer = d:sub(1, 1)
    if d:sub(2, 2) ~= outer then
        return {matched = false}
    end

    -- Due to palindrome: positions 6-7 (Lua) = 7-8 must also equal outer (auto)
    -- Inner bookend: positions 2-3 (Lua) must be the same digit ≠ outer
    local inner = d:sub(3, 3)
    if inner == outer then
        return {matched = false}
    end
    if d:sub(4, 4) ~= inner then
        return {matched = false}
    end

    -- Center 2 digits (positions 4-5, Lua) can be anything
    -- Already guaranteed by palindrome

    return {
        matched = true,
        group_boxes = {
            {from = 0, to = 1, color = "orange", thickness = 3},
            {from = 6, to = 7, color = "orange", thickness = 3},
            {from = 2, to = 3, color = "coral", thickness = 2},
            {from = 4, to = 5, color = "coral", thickness = 2},
        },
        connectors = {
            {from = 0, to = 7, color = "orange", style = "arc"},
            {from = 2, to = 5, color = "coral", style = "arc"},
        },
        message = outer .. outer .. " bookends + palindrome (CS-Bookend Full Radar CS-1280)"
    }
end
