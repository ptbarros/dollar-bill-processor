--[[
Pattern: SEVEN_IN_A_ROW
Description: 7 consecutive identical digits
Tier: 2
Examples: ["11111112", "27777777", "88888883"]
Odds: 1 in 555,556
Price: $80-$2,400
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local run = has_n_consecutive(digits, 7)
    if not run then
        return {matched = false}
    end

    -- Highlight the 7 consecutive digits
    local positions = {}
    for i = 0, 6 do
        table.insert(positions, run.start + i)
    end

    -- Find the odd digit out
    local odd_pos = nil
    for i = 0, 7 do
        local found = false
        for _, p in ipairs(positions) do
            if p == i then found = true; break end
        end
        if not found then
            odd_pos = i
            break
        end
    end

    return {
        matched = true,
        -- Single box around the 7 matching digits (Ed review), no per-digit boxes.
        highlights = {},
        group_boxes = {
            {from = positions[1], to = positions[#positions], color = "orange", thickness = 2}
        },
        connectors = {},
        message = "7 x " .. run.digit .. " in a row"
    }
end
