--[[
Pattern: QUADS
Description: 4 consecutive identical digits
Tier: 4
Examples: ["12333345", "11114567", "22229034"]
Odds: 1 in 217
Price: $3-$8 / $9-$20
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local run = has_n_consecutive(digits, 4)
    if not run then
        return {matched = false}
    end

    -- Highlight the 4 consecutive digits
    local positions = {}
    for i = 0, 3 do
        table.insert(positions, run.start + i)
    end

    return {
        matched = true,
        highlights = {
            highlight(positions, "gold", "quad")
        },
        connectors = {},
        message = "Quad " .. run.digit .. "s"
    }
end
