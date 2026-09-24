--[[
Pattern: FLIPPER
Description: Only flippable digits (0,1,6,8,9)
Tier: 8
Flippable: true
Examples: ["01689018", "96801896", "18906890"]
Odds: 1 in 279 (343,750 per 96M)
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

    -- Color-code each digit by how it transforms, matching Near Flipper (Ed review).
    local highlights = {}
    for i = 0, 7 do
        local d = digits:sub(i + 1, i + 1)
        local color
        if d == "0" or d == "8" then
            color = "purple"
        elseif d == "1" then
            color = "blue"
        else
            color = "magenta"
        end
        table.insert(highlights, highlight({i}, color, d))
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = {},
        message = "All flipper digits (0,1,6,8,9)"
    }
end
