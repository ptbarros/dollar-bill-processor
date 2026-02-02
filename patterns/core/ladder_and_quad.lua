--[[
Pattern: LADDER_AND_QUAD
Description: Contains 4+ ladder AND quad
Tier: 3
Examples: ["12343333", "44445678", "11112345"]
Odds: 1 in 140,350
Price: $40-$350+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Find ladder of 4+
    local ladder = find_ladder_of_length(digits, 4)
    if not ladder then
        return {matched = false}
    end

    -- Find quad
    local quad = has_n_consecutive(digits, 4)
    if not quad then
        return {matched = false}
    end

    -- Build ladder positions
    local ladder_pos = {}
    for i = 0, ladder.length - 1 do
        table.insert(ladder_pos, ladder.start + i)
    end

    -- Build quad positions
    local quad_pos = {}
    for i = 0, quad.length - 1 do
        table.insert(quad_pos, quad.start + i)
    end

    return {
        matched = true,
        highlights = {
            highlight(ladder_pos, "lime", "ladder"),
            highlight(quad_pos, "gold", "quad")
        },
        connectors = {},
        message = ladder.length .. "-ladder + quad " .. quad.digit .. "s"
    }
end
