--[[
Pattern: LUCKY_777
Description: Contains 777
Tier: 10
Examples: ["12377712", "77712345"]
Odds: 1 in 167
Price: $2-$5
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local pos = string.find(digits, "777", 1, true)
    if not pos then
        return {matched = false}
    end

    -- Highlight the 777 (convert to 0-indexed)
    local start_pos = pos - 1
    return {
        matched = true,
        highlights = {
            highlight({start_pos, start_pos + 1, start_pos + 2}, "gold", "lucky 777")
        },
        connectors = {},
        message = "Lucky 777"
    }
end
