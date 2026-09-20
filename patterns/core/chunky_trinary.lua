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

    -- One colored box per run (Ed review): removed the individual digit highlights.
    local colors = {"blue", "orange", "magenta"}
    local group_boxes = {}
    for i, run in ipairs(runs) do
        table.insert(group_boxes, {from = run.start, to = run.start + run.length - 1, color = colors[i], thickness = 3})
    end

    return {
        matched = true,
        highlights = {},
        group_boxes = group_boxes,
        connectors = {},
        message = "Chunky trinary"
    }
end
