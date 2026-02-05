--[[
Pattern: NICKS_FULL_HOUSE
DisplayName: Full House
Description: Three of a kind followed by a pair (AAABB) or pair followed by three (AABBB)
Tier: 5
Examples: ["11122345", "12333456", "00011234", "99988765"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Pattern 1: AAABB (3 same, then 2 same)
    for start = 1, 4 do
        local seg = s:sub(start, start + 4)
        if seg:sub(1, 1) == seg:sub(2, 2) and seg:sub(2, 2) == seg:sub(3, 3) and
           seg:sub(4, 4) == seg:sub(5, 5) and seg:sub(3, 3) ~= seg:sub(4, 4) then
            return {
                matched = true,
                message = "Full house: " .. seg:sub(1, 1) .. seg:sub(1, 1) .. seg:sub(1, 1) .. " + " .. seg:sub(4, 4) .. seg:sub(4, 4),
                group_boxes = {
                    {from = start - 1, to = start + 1, color = "orange"},
                    {from = start + 2, to = start + 3, color = "cyan"}
                }
            }
        end
    end

    -- Pattern 2: AABBB (2 same, then 3 same)
    for start = 1, 4 do
        local seg = s:sub(start, start + 4)
        if seg:sub(1, 1) == seg:sub(2, 2) and
           seg:sub(3, 3) == seg:sub(4, 4) and seg:sub(4, 4) == seg:sub(5, 5) and
           seg:sub(2, 2) ~= seg:sub(3, 3) then
            return {
                matched = true,
                message = "Full house: " .. seg:sub(1, 1) .. seg:sub(1, 1) .. " + " .. seg:sub(3, 3) .. seg:sub(3, 3) .. seg:sub(3, 3),
                group_boxes = {
                    {from = start - 1, to = start, color = "cyan"},
                    {from = start + 1, to = start + 3, color = "orange"}
                }
            }
        end
    end

    return {matched = false}
end
