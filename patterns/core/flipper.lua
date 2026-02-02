--[[
Pattern: FLIPPER
Description: Only flippable digits (0,1,6,8,9)
Tier: 8
Examples: ["01689018", "96801896", "18906890"]
Odds: 1 in 256
Price: $3-$10
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check all digits are flip-valid (0, 1, 6, 8, 9)
    if not all_flip_valid(digits) then
        return {matched = false}
    end

    -- Highlight all positions in purple
    local positions = {0, 1, 2, 3, 4, 5, 6, 7}

    return {
        matched = true,
        highlights = {
            highlight(positions, "purple", "flipper")
        },
        connectors = {},
        message = "All flipper digits (0,1,6,8,9)"
    }
end
