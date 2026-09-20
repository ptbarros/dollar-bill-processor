--[[
Pattern: CHUNKY_TRINARY
Description: Three different digits, each in one unbroken block, with at least one appearing only once (e.g. 0·55555·22). If every block is two or more, it's a Super Trinary instead.
Tier: 4
Examples: ["05555522", "13333355", "88800001", "44999992"]
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

    -- At least one digit appears only once (a lone single). This is what makes it a
    -- Chunky Trinary rather than a Super Trinary, whose blocks are all 2+ (Ed review).
    local has_single = false
    for _, run in ipairs(runs) do
        if run.length == 1 then has_single = true break end
    end
    if not has_single then return {matched = false} end

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
