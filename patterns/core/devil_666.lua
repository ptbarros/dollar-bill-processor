--[[
Pattern: DEVIL_666
Description: Contains 666
Tier: 10
Examples: ["12366612", "66612345"]
Odds: 1 in 167
Price: $2-$5
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local pos = string.find(digits, "666", 1, true)
    if not pos then
        return {matched = false}
    end

    -- Highlight the 666 (convert to 0-indexed)
    local start_pos = pos - 1
    return {
        matched = true,
        highlights = {
            highlight({start_pos, start_pos + 1, start_pos + 2}, "red", "devil 666")
        },
        connectors = {},
        message = "Devil 666"
    }
end
