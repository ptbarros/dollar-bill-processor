--[[
Pattern: NICKS_CHUNKY_LADDER_5
DisplayName: 5 Digit Chunky Ladder
Description: 5 consecutive digits (a 5-rung ladder) in sorted order, each appearing one or more times (e.g. 01122334).
Tier: 5
Odds: 1 in 249,351 (385 per 96M)
Examples: ["01122334", "44332210", "00112234", "43322110"]
--]]

function match(ctx)
    local s = ctx.digits

    local seen = {}
    for i = 1, 8 do seen[s:sub(i, i)] = true end
    local uniq = {}
    for ch in pairs(seen) do table.insert(uniq, tonumber(ch)) end
    if #uniq ~= 5 then return {matched = false} end

    -- The 5 unique digits must be consecutive (a real ladder), not just any 5.
    table.sort(uniq)
    for i = 1, 4 do
        if uniq[i + 1] - uniq[i] ~= 1 then return {matched = false} end
    end

    -- Digits must be in sorted order (ascending or descending).
    local sorted = {}
    for i = 1, 8 do table.insert(sorted, s:sub(i, i)) end
    table.sort(sorted)
    local dir
    if table.concat(sorted) == s then
        dir = "ascending"
    else
        table.sort(sorted, function(a, b) return a > b end)
        if table.concat(sorted) == s then
            dir = "descending"
        else
            return {matched = false}
        end
    end

    -- One colored box per run of identical digits (like the 3-digit version, Ed review).
    local colors = {"blue", "orange", "magenta", "red", "purple"}
    local boxes = {}
    for _, run in ipairs(find_runs(s)) do
        table.insert(boxes, {from = run.start, to = run.start + run.length - 1,
            color = colors[(#boxes % #colors) + 1], thickness = 3})
    end

    return {
        matched = true,
        message = "5-digit chunky ladder (" .. dir .. ")",
        highlights = {},
        group_boxes = boxes
    }
end
