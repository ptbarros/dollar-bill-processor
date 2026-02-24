--[[
Pattern: CS_LEADING_CENTER_TRAILING_QUADS
DisplayName: CS-Leading, Center & Trailing Quads
Description: Any four of the same digit grouped consecutively anywhere within the serial. Leading Quads start at position 1 (e.g., M 4444xxxx M), Center Quads start at positions 2, 3, or 4 (e.g., M x4444xxx M, M xx4444xx M, M xxx4444x M), Trailing Quads start at position 5 (e.g., M xxxx4444 M). This is essentially the same as CS-Quad (CS-200) — any grouped run of 4+ same digit.
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
