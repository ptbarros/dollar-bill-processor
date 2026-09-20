--[[
Pattern: NICKS_COUNTING_LADDER
DisplayName: Alternator Ladder
Description: Four 2-digit numbers counting up or down by 1 (e.g. 12 13 14 15, or 18 19 20 21). Each value 10-96.
Tier: 4
Examples: ["12131415", "18192021", "94939291"]
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Four 2-digit numbers
    local nums = {}
    for i = 1, 4 do
        nums[i] = tonumber(d:sub((i - 1) * 2 + 1, i * 2))
    end

    -- Each value must sit in the real serial range (10-96 million); this also keeps
    -- out leading-zero pairs like 01 02 03 04.
    for i = 1, 4 do
        if nums[i] < 10 or nums[i] > 96 then return {matched = false} end
    end

    -- They must count by exactly 1, all up or all down (numeric, so 19->20 counts).
    local is_asc, is_desc = true, true
    for i = 1, 3 do
        if nums[i + 1] - nums[i] ~= 1 then is_asc = false end
        if nums[i] - nums[i + 1] ~= 1 then is_desc = false end
    end
    if not is_asc and not is_desc then return {matched = false} end

    local direction = is_asc and "ascending" or "descending"
    return {
        matched = true,
        message = string.format("2-digit counting ladder %s: %d %d %d %d",
            direction, nums[1], nums[2], nums[3], nums[4]),
        -- One group box per 2-digit pair (Ed review).
        highlights = {},
        group_boxes = {
            {from = 0, to = 1, color = "blue", thickness = 3},
            {from = 2, to = 3, color = "orange", thickness = 3},
            {from = 4, to = 5, color = "magenta", thickness = 3},
            {from = 6, to = 7, color = "red", thickness = 3}
        }
    }
end
