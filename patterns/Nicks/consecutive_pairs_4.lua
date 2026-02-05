--[[
Pattern: CONSECUTIVE_PAIRS_4
DisplayName: 4 Consecutive Pairs
Description: AABBCCDD pattern (4 consecutive pairs of identical digits)
Tier: 4
Examples: ["11223344", "00112233", "99887766", "55667788"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Check each pair position
    for i = 1, 8, 2 do
        local pair = s:sub(i, i + 1)
        if pair:sub(1, 1) ~= pair:sub(2, 2) then
            return {matched = false}
        end
    end

    return {
        matched = true,
        message = "4 consecutive pairs",
        group_boxes = {
            {from = 0, to = 1, color = "orange"},
            {from = 2, to = 3, color = "cyan"},
            {from = 4, to = 5, color = "lime"},
            {from = 6, to = 7, color = "gold"}
        }
    }
end
