--[[
Pattern: CS_TRIPLE_TRIPLE_PAIR
DisplayName: CS-Triple-Triple-Pair
Description: Two CS-Triples (runs of 3+) plus at least one CS-Pair (run of 2), all as consecutive groups.
BookRef: CS-130
Tier: 5
Examples: ["00033300", "11122200", "11100022"]
Odds: 1 in 27,720
Price: $20-$60
--]]

function match(ctx)
    local runs = find_runs(ctx.digits)

    local triple_runs = {}
    local pair_runs = {}

    for _, run in ipairs(runs) do
        if run.length >= 3 then
            table.insert(triple_runs, run)
        elseif run.length == 2 then
            table.insert(pair_runs, run)
        end
    end

    if #triple_runs < 2 or #pair_runs < 1 then
        return {matched = false}
    end

    local group_boxes = {}
    local colors = {"gold", "coral", "cyan", "lime"}

    for i, run in ipairs(triple_runs) do
        table.insert(group_boxes, {
            from = run.start,
            to = run.start + run.length - 1,
            color = colors[((i - 1) % #colors) + 1],
            thickness = 2
        })
    end
    for i, run in ipairs(pair_runs) do
        table.insert(group_boxes, {
            from = run.start,
            to = run.start + run.length - 1,
            color = "salmon",
            thickness = 2
        })
    end

    local msg = #triple_runs .. " triples + " .. #pair_runs .. " pair(s)"
    return {
        matched = true,
        group_boxes = group_boxes,
        message = msg
    }
end
