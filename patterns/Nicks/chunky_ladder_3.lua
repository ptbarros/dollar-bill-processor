--[[
Pattern: NICKS_CHUNKY_LADDER_3
DisplayName: 3 Digit Chunky Ladder
Description: 3 unique digits all in sorted order (ascending or descending)
Tier: 7
Examples: ["00000123", "01222222", "98700000", "33332100"]
--]]

function match(ctx)
    local s = ctx.digits

    local unique = {}
    for i = 1, 8 do
        unique[s:sub(i, i)] = true
    end

    local count = 0
    for _ in pairs(unique) do count = count + 1 end

    if count ~= 3 then
        return {matched = false}
    end

    local sorted_asc = {}
    for i = 1, 8 do
        table.insert(sorted_asc, s:sub(i, i))
    end
    table.sort(sorted_asc)

    local dir
    if table.concat(sorted_asc) == s then
        dir = "ascending"
    else
        table.sort(sorted_asc, function(a, b) return a > b end)
        if table.concat(sorted_asc) == s then
            dir = "descending"
        else
            return {matched = false}
        end
    end

    -- One colored box per run of identical digits (Ed review), e.g. 9999 888 7.
    local colors = {"blue", "orange", "magenta", "red"}
    local boxes = {}
    for _, run in ipairs(find_runs(s)) do
        table.insert(boxes, {from = run.start, to = run.start + run.length - 1,
            color = colors[(#boxes % #colors) + 1], thickness = 3})
    end
    return {
        matched = true,
        message = "3-digit chunky ladder (" .. dir .. ")",
        highlights = {},
        group_boxes = boxes
    }
end
