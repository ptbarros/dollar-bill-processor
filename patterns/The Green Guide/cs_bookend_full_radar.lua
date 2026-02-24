--[[
Pattern: CS_BOOKEND_FULL_RADAR
DisplayName: CS-Bookend Full Radar
Description: A CS-Full Radar (palindrome) where the first two digits are identical (forming a CS-Pair bookend at each end): e.g., M 44133144 M. By the palindrome, positions 6-7 automatically mirror positions 0-1.
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

    -- Positions 6-7 also equal outer automatically (palindrome guarantee)

    return {
        matched = true,
        group_boxes = {
            {from = 0, to = 1, color = "orange", thickness = 3},
            {from = 6, to = 7, color = "orange", thickness = 3},
            {from = 2, to = 5, color = "coral", thickness = 2},
        },
        connectors = {
            {from = 0, to = 7, color = "orange", style = "arc"},
            {from = 1, to = 6, color = "orange", style = "arc"},
        },
        message = outer .. outer .. " bookends + palindrome (CS-Bookend Full Radar CS-1280)"
    }
end
