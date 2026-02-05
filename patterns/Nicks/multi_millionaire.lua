--[[
Pattern: NICKS_MULTI_MILLIONAIRE
DisplayName: Multi Millionaire
Description: Round millions (10000000, 20000000, etc.)
Tier: 2
Examples: ["10000000", "20000000", "50000000", "90000000"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Pattern: [1-9]0000000
    if s:match("^[1-9]0000000$") then
        local millions = tonumber(s:sub(1, 1)) * 10
        return {
            matched = true,
            message = millions .. " million",
            highlights = {
                {positions = {0}, color = "gold"},
                {positions = {1,2,3,4,5,6,7}, color = "gray"}
            }
        }
    end

    return {matched = false}
end
