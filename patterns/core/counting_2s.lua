--[[
Pattern: COUNTING_2S
Description: Counting by 2s ladder (02040608)
Tier: 7
Examples: ["02040608", "12141618"]
Odds: 1 in 1,116,279
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

    -- Check if they form arithmetic sequence with step 2
    local valid = true
    for i = 1, 3 do
        if nums[i+1] - nums[i] ~= 2 then
            valid = false
            break
        end
    end

    if not valid then
        return {matched = false}
    end

    return {
        matched = true,
        highlights = {
            highlight({0, 1}, "lime", "first"),
            highlight({2, 3}, "teal", "second"),
            highlight({4, 5}, "cyan", "third"),
            highlight({6, 7}, "blue", "fourth")
        },
        group_boxes = {
            {from = 0, to = 1, color = "lime", thickness = 2},
            {from = 2, to = 3, color = "teal", thickness = 2},
            {from = 4, to = 5, color = "cyan", thickness = 2},
            {from = 6, to = 7, color = "blue", thickness = 2}
        },
        connectors = {},
        message = string.format("Counting by 2s: %02d->%02d->%02d->%02d", nums[1], nums[2], nums[3], nums[4])
    }
end
