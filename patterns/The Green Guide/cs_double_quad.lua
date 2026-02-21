--[[
Pattern: CS_DOUBLE_QUAD
DisplayName: CS-Double Quad
Description: Two separate groups of 4+ consecutive identical digits. Both quads must be grouped (consecutive). e.g., M 11114444 M.
BookRef: CS-230
Tier: 3
Examples: ["11114444", "44441111", "33335555"]
Odds: 1 in 1,111,110
Price: $25-$100
--]]

function match(ctx)
    local d = ctx.digits
    local runs = find_runs(d)

    -- Find all runs of 4+ digits
    local quads = {}
    for _, run in ipairs(runs) do
        if run.length >= 4 then
            table.insert(quads, run)
        end
    end

    -- Must have at least 2 separate runs of 4+
    if #quads < 2 then
        return {matched = false}
    end

    -- Build group boxes for each quad run
    local group_boxes = {}
    local colors = {"gold", "orange"}
    for i, quad in ipairs(quads) do
        local color = colors[((i - 1) % #colors) + 1]
        table.insert(group_boxes, {
            from = quad.start,
            to = quad.start + quad.length - 1,
            color = color,
            thickness = 3
        })
    end

    local msg = quads[1].length .. " " .. quads[1].digit .. "s + " ..
                quads[2].length .. " " .. quads[2].digit .. "s (CS-Double Quad)"

    return {
        matched = true,
        group_boxes = group_boxes,
        message = msg
    }
end
