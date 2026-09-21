--[[
Pattern: CS_TRIPLE_DOUBLE_DOUBLE
DisplayName: CS-Triple Double Double
Description: A solid block of three identical digits plus two side-by-side pairs of other digits (e.g. 111·22·33).
BookRef: CS-170
Tier: 5
Examples: ["11100223", "11122334", "44422300"]
Price: $15-$40
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

    if #triple_runs < 1 or #pair_runs < 2 then
        return {matched = false}
    end

    local group_boxes = {}

    for i, run in ipairs(triple_runs) do
        table.insert(group_boxes, {
            from = run.start,
            to = run.start + run.length - 1,
            color = "gold",
            thickness = 2
        })
    end

    local pair_colors = {"coral", "cyan"}
    for i, run in ipairs(pair_runs) do
        table.insert(group_boxes, {
            from = run.start,
            to = run.start + run.length - 1,
            color = pair_colors[((i - 1) % #pair_colors) + 1],
            thickness = 2
        })
    end

    local msg = #triple_runs .. " triple(s) + " .. #pair_runs .. " pairs"
    return {
        matched = true,
        group_boxes = group_boxes,
        message = msg
    }
end
