--[[
Pattern: TWIN_SET_DOUBLES
DisplayName: Twin Sets Doubles
Description: Pattern ABABCDCD where positions 1&3, 2&4, 5&7, 6&8 match
Tier: 5
Examples: ["12123434", "00110022", "98987676", "12124545"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Check pairs: 1&3, 2&4, 5&7, 6&8
    if s:sub(1, 1) ~= s:sub(3, 3) then return {matched = false} end
    if s:sub(2, 2) ~= s:sub(4, 4) then return {matched = false} end
    if s:sub(5, 5) ~= s:sub(7, 7) then return {matched = false} end
    if s:sub(6, 6) ~= s:sub(8, 8) then return {matched = false} end

    return {
        matched = true,
        message = "Twin set doubles: " .. s:sub(1, 2) .. "×2 + " .. s:sub(5, 6) .. "×2",
        group_boxes = {
            {from = 0, to = 1, color = "red"},
            {from = 2, to = 3, color = "red"},
            {from = 4, to = 5, color = "blue"}, 
            {from = 6, to = 7, color = "blue"}
        },
        
    }
end
