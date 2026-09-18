--[[
Pattern: SIX_IN_A_ROW
Description: 6 consecutive identical digits
Tier: 3
Examples: ["66666612", "23333339", "11111189"]
Odds: 1 in 35,556
Price: $25-$150
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local run = has_n_consecutive(digits, 6)
    if not run then
        return {matched = false}
    end

    -- Highlight the 6 consecutive digits
    local positions = {}
    for i = 0, 5 do
        table.insert(positions, run.start + i)
    end

    return {
        matched = true,
        -- Single box around the run (Ed review), no per-digit boxes.
        highlights = {},
        group_boxes = {
            {from = positions[1], to = positions[#positions], color = "orange", thickness = 2}
        },
        connectors = {},
        message = "6 x " .. run.digit .. " in a row"
    }
end
