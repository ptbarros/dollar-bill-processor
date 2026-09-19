--[[
Pattern: NICKS_SUPER_LADDER
DisplayName: Super Ladder
Description: 3 unique digits with specific counts (3-3-2, 4-4-2, or 2-2-4), all sorted
Tier: 4
Examples: ["00011122", "00112233", "11222233", "33322211", "22211100"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Count unique digits
    local unique = {}
    for i = 1, 8 do
        unique[s:sub(i, i)] = true
    end

    local count = 0
    for _ in pairs(unique) do count = count + 1 end

    if count ~= 3 then
        return {matched = false}
    end

    -- Count occurrences
    local counts = {}
    for i = 1, 8 do
        local d = s:sub(i, i)
        counts[d] = (counts[d] or 0) + 1
    end

    -- Get sorted counts
    local count_list = {}
    for _, c in pairs(counts) do
        table.insert(count_list, c)
    end
    table.sort(count_list)

    -- Valid combinations: [2,3,3], [2,2,4], [2,4,4] (sorted)
    local valid = false
    if count_list[1] == 2 and count_list[2] == 3 and count_list[3] == 3 then
        valid = true
    elseif count_list[1] == 2 and count_list[2] == 2 and count_list[3] == 4 then
        valid = true
    elseif count_list[1] == 2 and count_list[2] == 4 and count_list[3] == 4 then
        valid = true -- Not in original but similar pattern
    end

    if not valid then
        return {matched = false}
    end

    -- Check if sorted
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

    -- One colored box per run of identical digits (Ed review).
    local colors = {"blue", "orange", "magenta", "red"}
    local boxes = {}
    for _, run in ipairs(find_runs(s)) do
        table.insert(boxes, {from = run.start, to = run.start + run.length - 1,
            color = colors[(#boxes % #colors) + 1], thickness = 3})
    end
    return {
        matched = true,
        message = "Super ladder (" .. dir .. ")",
        highlights = {},
        group_boxes = boxes
    }
end
