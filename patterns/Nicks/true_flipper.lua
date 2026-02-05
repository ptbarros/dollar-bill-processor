--[[
Pattern: NICKS_TRUE_FLIPPER
DisplayName: True Flipper
Description: Only digits 0, 6, 9 used (looks same upside down)
Tier: 4
Examples: ["06960696", "00669900", "96069606", "99006699"]
--]]

function match(ctx)
    local s = ctx.digits

    for i = 1, 8 do
        local d = s:sub(i, i)
        if d ~= "0" and d ~= "6" and d ~= "9" then
            return {matched = false}
        end
    end

    return {
        matched = true,
        message = "True flipper: only 0, 6, 9",
        highlights = {{positions = {0,1,2,3,4,5,6,7}, color = "magenta"}}
    }
end
