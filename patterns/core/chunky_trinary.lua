--[[
Pattern: CHUNKY_TRINARY
Description: Trinary with chunked digits
Tier: 4
Examples: ["11133355", "22244466"]
Odds: 1 in 6,666
Price: $10-$50+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Must be trinary
    if unique_count(digits) ~= 3 then
        return {matched = false}
    end

    -- Check for chunked pattern - each digit appears in consecutive runs
    local runs = find_runs(digits)

    -- Should have exactly 3 runs (one for each unique digit)
    if #runs ~= 3 then
        return {matched = false}
    end

    -- Each run should be at least 2 digits
    for _, run in ipairs(runs) do
        if run.length < 2 then
            return {matched = false}
        end
    end

    local colors = {"lime", "teal", "cyan"}
    local highlights = {}
    local group_boxes = {}

    for i, run in ipairs(runs) do
        local positions = {}
        for j = 0, run.length - 1 do
            table.insert(positions, run.start + j)
        end
        table.insert(highlights, highlight(positions, colors[i], run.digit))
        table.insert(group_boxes, {from = run.start, to = run.start + run.length - 1, color = colors[i], thickness = 2})
    end

    return {
        matched = true,
        highlights = highlights,
        group_boxes = group_boxes,
        connectors = {},
        message = "Chunky trinary"
    }
end
