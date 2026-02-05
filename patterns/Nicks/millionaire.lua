--[[
Pattern: NICKS_MILLIONAIRE
DisplayName: Millionaire
Description: Round millions with any leading digits (XX000000)
Tier: 3
Examples: ["01000000", "12000000", "99000000", "55000000"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Pattern: XX000000 where first two digits form 01-99
    if s:match("^%d%d000000$") then
        local millions = tonumber(s:sub(1, 2))
        if millions >= 1 then
            return {
                matched = true,
                message = millions .. " million",
                highlights = {
                    {positions = {0, 1}, color = "gold"},
                    {positions = {2,3,4,5,6,7}, color = "gray"}
                }
            }
        end
    end

    return {matched = false}
end
