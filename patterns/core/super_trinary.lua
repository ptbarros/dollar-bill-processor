--[[
Pattern: SUPER_TRINARY
Description: Trinary with structured groups (AAACCCEE, etc.)
Tier: 3
Examples: ["11133355", "22244466", "00022244"]
Odds: 1 in 24,691
Price: $20-$100+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Must be trinary (exactly 3 unique digits)
    if unique_count(digits) ~= 3 then
        return {matched = false}
    end

    -- Check for pattern where each digit appears in groups (2+ consecutive)
    -- Look for structured groupings like AAA BBB CC or AA BBB CCC
    local runs = find_runs(digits)

    -- Check if we have 3 or fewer runs (meaning digits are grouped)
    if #runs > 4 then
        return {matched = false}
    end

    -- Check if each run is at least 2 long
    local structured = true
    for _, run in ipairs(runs) do
        if run.length < 2 then
            structured = false
            break
        end
    end

    if not structured then
        return {matched = false}
    end

    -- Highlight each group
    local colors = {"lime", "teal", "cyan", "blue"}
    local highlights = {}
    local group_boxes = {}

    for i, run in ipairs(runs) do
        local positions = {}
        for j = 0, run.length - 1 do
            table.insert(positions, run.start + j)
        end
        table.insert(highlights, highlight(positions, colors[i] or "gray", "group"))
        table.insert(group_boxes, {from = run.start, to = run.start + run.length - 1, color = colors[i] or "gray", thickness = 2})
    end

    return {
        matched = true,
        highlights = highlights,
        group_boxes = group_boxes,
        connectors = {},
        message = "Super trinary (structured groups)"
    }
end
