--[[
Pattern: NICKS_CHUNKY_LADDER_4
DisplayName: 4 Digit Chunky Ladder
Description: Four consecutive digits (like 3-4-5-6), each in its own chunk, the whole serial sorted up or down (e.g. 00·11·22·33).
Tier: 3
Odds: 1 in 210,989 (455 per 96M)
Examples: ["00112233", "01112223", "33221100", "66778899"]
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
    if count ~= 4 then
        return {matched = false}
    end

    -- The 4 distinct digits must be consecutive on the number line (a real
    -- ladder, not just any 4 sorted digits) -- Ed review.
    if hi - lo ~= 3 then
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

    -- One colored box per run of identical digits (Ed review), e.g. 00 11 22 33.
    local colors = {"blue", "orange", "magenta", "red", "purple", "hotpink", "black"}
    local boxes = {}
    for _, run in ipairs(find_runs(s)) do
        table.insert(boxes, {from = run.start, to = run.start + run.length - 1,
            color = colors[(#boxes % #colors) + 1], thickness = 3})
    end

    return {
        matched = true,
        message = "4-digit chunky ladder (" .. dir .. ")",
        highlights = {},
        group_boxes = boxes
    }
end
