--[[
Pattern: COUNTING_5S
DisplayName: Counting By 5s Ladder
Description: Counting by 5s ladder (05101520)
Tier: 7
Examples: ["05101520", "10152025"]
Odds: 1 in 1,129,412 (85 per 96M)
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

    -- Check if they form arithmetic sequence with step 5
    local valid = true
    for i = 1, 3 do
        if nums[i+1] - nums[i] ~= 5 then
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
        message = string.format("Counting by 5s: %02d->%02d->%02d->%02d", nums[1], nums[2], nums[3], nums[4])
    }
end
