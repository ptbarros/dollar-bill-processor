--[[
Pattern: CS_DOUBLE_TRIPLES
DisplayName: CS-Double Triples
Description: Two separate CS-Triples (runs of 3+ identical digits) anywhere in the serial.
BookRef: CS-160
Tier: 5
Examples: ["00011100", "11100222", "00033355"]
Price: $10-$30
--]]

function match(ctx)
    local runs = find_runs(ctx.digits)

    local triple_runs = {}
    for _, run in ipairs(runs) do
        if run.length >= 3 then
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

    local msg = #triple_runs .. " triple groups"
    return {
        matched = true,
        group_boxes = group_boxes,
        message = msg
    }
end
