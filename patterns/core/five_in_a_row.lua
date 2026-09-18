--[[
Pattern: FIVE_IN_A_ROW
Description: 5 consecutive identical digits
Tier: 4
Examples: ["55555123", "38177777", "12333334"]
Odds: 1 in 2,702
Price: $8-$50
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local run = has_n_consecutive(digits, 5)
    if not run then
        return {matched = false}
    end

    -- Highlight the 5 consecutive digits
    local positions = {}
    for i = 0, 4 do
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
        message = "5 x " .. run.digit .. " in a row"
    }
end
