--[[
Pattern: CS_LEADING_CENTER_TRAILING_QUADS
DisplayName: CS-Leading, Center & Trailing Quads
Description: A solid block of four identical digits sitting anywhere in the serial — at the front, the middle, or the end (e.g. 4444·xxxx).
BookRef: CS-220
Tier: 4
Examples: ["44441234", "12444412", "12344444"]
Odds: 1 in 11,112
Price: $10-$50
--]]

function match(ctx)
    local d = ctx.digits
    local runs = find_runs(d)

    for _, run in ipairs(runs) do
        if run.length >= 4 then
            -- Determine position label
            local start_pos = run.start + 1  -- 1-indexed for labeling
            local label
            if start_pos == 1 then
                label = "Leading Quad"
            elseif start_pos + run.length - 1 == 8 then
                label = "Trailing Quad"
            else
                label = "Center Quad"
            end

            return {
                matched = true,
                group_boxes = {
                    {from = run.start, to = run.start + run.length - 1, color = "gold", thickness = 3}
                },
                message = run.digit .. "x" .. run.length .. " " .. label .. " at pos " .. start_pos .. " (CS-Leading/Center/Trailing Quads)"
            }
        end
    end

    return {matched = false}
end
