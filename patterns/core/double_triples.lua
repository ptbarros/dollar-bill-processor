--[[
Pattern: DOUBLE_TRIPLES
DisplayName: Two Triple Runs
Description: Two or more separate runs of exactly three identical digits, with no run of four or more anywhere (e.g. 444·6·7·000, where 444 and 000 both qualify).
BookRef: CS-150
Tier: 5
Odds: 1 in 2,232 (43,011 per 96M)
Examples: ["44467000", "00011100", "11100222", "00033355"]
Price: $10-$30
--]]

function match(ctx)
    local runs = find_runs(ctx.digits)

    -- Reject any serial that has a run of four or more identical digits (Ed review).
    for _, run in ipairs(runs) do
        if run.length >= 4 then return {matched = false} end
    end

    -- Collect runs of exactly three.
    local triple_runs = {}
    for _, run in ipairs(runs) do
        if run.length == 3 then
            table.insert(triple_runs, run)
        end
    end

    if #triple_runs < 2 then
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

    return {
        matched = true,
        group_boxes = group_boxes,
        message = #triple_runs .. " triple runs"
    }
end
