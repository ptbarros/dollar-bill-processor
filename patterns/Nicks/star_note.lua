--[[
Pattern: STAR_NOTE
DisplayName: Star Note
Description: Serial ends with * (replacement note)
Tier: 7
Examples: ["12345678"]
--]]

function match(ctx)
    local full = ctx.full_serial

    if full:sub(-1) == "*" then
        return {
            matched = true,
            message = "Star note (replacement)",
            highlights = {{positions = {0,1,2,3,4,5,6,7}, color = "gold"}}
        }
    end

    return {matched = false}
end
