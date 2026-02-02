--[[
Pattern: TRIPLE_DOUBLE_DOUBLE
Description: Triple + double + double pattern
Tier: 4
Examples: ["11122334", "33344556"]
Odds: 1 in 2,469
Price: $5-$25
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Look for patterns like AAABBCCD or AABBCCCD
    local runs = find_runs(digits)

    -- Need exactly 4 runs
    if #runs ~= 4 then
        return {matched = false}
    end

    -- Count run lengths
    local triple_count = 0
    local double_count = 0
    local single_count = 0

    for _, run in ipairs(runs) do
        if run.length == 3 then
            triple_count = triple_count + 1
        elseif run.length == 2 then
            double_count = double_count + 1
        elseif run.length == 1 then
            single_count = single_count + 1
        end
    end

    -- Must have 1 triple and 2 doubles (plus 1 single for the 8th digit)
    if triple_count ~= 1 or double_count ~= 2 or single_count ~= 1 then
        return {matched = false}
    end

    local colors = {"gold", "coral", "magenta", "purple"}
    local highlights = {}
    local group_boxes = {}

    for i, run in ipairs(runs) do
        local positions = {}
        for j = 0, run.length - 1 do
            table.insert(positions, run.start + j)
        end
        table.insert(highlights, highlight(positions, colors[i], "group"))
        if run.length >= 2 then
            table.insert(group_boxes, {from = run.start, to = run.start + run.length - 1, color = colors[i], thickness = 2})
        end
    end

    return {
        matched = true,
        highlights = highlights,
        group_boxes = group_boxes,
        connectors = {},
        message = "Triple + double + double"
    }
end
