--[[
Pattern: COUNTING_8S
DisplayName: Counting By 8s Ladder
Description: Counting by 8s ladder (08162432)
Tier: 7
Examples: ["08162432", "16243240"]
Odds: 1 in 1,263,158 (76 per 96M)
Price: $100-$1,000+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Parse as four 2-digit numbers
    local nums = {}
    for i = 1, 4 do
        local pair = digits:sub((i-1)*2 + 1, i*2)
        table.insert(nums, tonumber(pair))
    end

    -- Check if they form arithmetic sequence with step 8
    local valid = true
    for i = 1, 3 do
        if nums[i+1] - nums[i] ~= 8 then
            valid = false
            break
        end
    end

    if not valid then
        return {matched = false}
    end

    return {
        matched = true,
        -- Per-digit inner boxes removed (Ed review): each 2-digit pair is shown
        -- by its single group box only, no arcs.
        highlights = {},
        group_boxes = {
            {from = 0, to = 1, color = "blue", thickness = 3},
            {from = 2, to = 3, color = "orange", thickness = 3},
            {from = 4, to = 5, color = "magenta", thickness = 3},
            {from = 6, to = 7, color = "red", thickness = 3}
        },
        connectors = {},
        message = string.format("Counting by 8s: %02d->%02d->%02d->%02d", nums[1], nums[2], nums[3], nums[4])
    }
end
