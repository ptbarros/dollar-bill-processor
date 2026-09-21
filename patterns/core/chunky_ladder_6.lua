--[[
Pattern: NICKS_CHUNKY_LADDER_6
DisplayName: 6 Digit Chunky Ladder
Description: Six consecutive digits, each in its own chunk, the whole serial sorted up or down (e.g. 00·1·2·3·4·55).
Tier: 3
Odds: 1 in 507,937 (189 per 96M)
Examples: ["00123455", "11234566", "55443210", "99876654"]
--]]

function match(ctx)
    local s = ctx.digits

    local unique = {}
    local lo, hi = 9, 0
    for i = 1, 8 do
        local d = s:sub(i, i)
        unique[d] = true
        local n = tonumber(d)
        if n < lo then lo = n end
        if n > hi then hi = n end
    end

    local count = 0
    for _ in pairs(unique) do count = count + 1 end
    if count ~= 6 then
        return {matched = false}
    end

    -- The 6 distinct digits must be consecutive on the number line (a real
    -- ladder, not just any 6 sorted digits) -- Ed review.
    if hi - lo ~= 5 then
        return {matched = false}
    end

    -- Whole serial must run in one direction (all sorted ascending or descending).
    local chars = {}
    for i = 1, 8 do chars[i] = s:sub(i, i) end
    table.sort(chars)
    local dir
    if table.concat(chars) == s then
        dir = "ascending"
    else
        table.sort(chars, function(a, b) return a > b end)
        if table.concat(chars) == s then
            dir = "descending"
        else
            return {matched = false}
        end
    end

    -- One colored box per run of identical digits (Ed review).
    local colors = {"blue", "orange", "magenta", "red", "purple", "hotpink", "black"}
    local boxes = {}
    for _, run in ipairs(find_runs(s)) do
        table.insert(boxes, {from = run.start, to = run.start + run.length - 1,
            color = colors[(#boxes % #colors) + 1], thickness = 3})
    end

    return {
        matched = true,
        message = "6-digit chunky ladder (" .. dir .. ")",
        highlights = {},
        group_boxes = boxes
    }
end
