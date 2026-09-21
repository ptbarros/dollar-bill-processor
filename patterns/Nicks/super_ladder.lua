--[[
Pattern: NICKS_SUPER_LADDER
DisplayName: Super Ladder
Description: Three consecutive digits in sorted order, with counts like 3-3-2 or 4-2-2 (e.g. 000·111·22).
Tier: 4
Odds: 1 in 1,066,667 (90 per 96M)
Examples: ["00011122", "11222233", "00001122", "33322211", "22211100"]
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

    -- The 3 distinct digits must be consecutive on the number line (Ed review):
    -- a real ladder, e.g. 3-4-5, not any three sorted digits.
    local lo, hi = 9, 0
    for d in pairs(unique) do
        local n = tonumber(d)
        if n < lo then lo = n end
        if n > hi then hi = n end
    end
    if hi - lo ~= 2 then
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

    -- Valid count multisets (sorted): [2,3,3] covers 3-3-2/3-2-3/2-3-3;
    -- [2,2,4] covers 4-2-2/2-4-2/2-2-4. (The old [2,4,4] summed to 10 -- dead.)
    local valid = false
    if count_list[1] == 2 and count_list[2] == 3 and count_list[3] == 3 then
        valid = true
    elseif count_list[1] == 2 and count_list[2] == 2 and count_list[3] == 4 then
        valid = true
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
