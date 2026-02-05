--[[
Pattern: NICKS_TRUE_BINARY
DisplayName: True Binary
Description: Only digits 0 and 1 used
Tier: 3
Examples: ["01010101", "00001111", "10101010", "11110000"]
--]]

function match(ctx)
    local s = ctx.digits

    for i = 1, 8 do
        local d = s:sub(i, i)
        if d ~= "0" and d ~= "1" then
            return {matched = false}
        end
    end

    return {
        matched = true,
        message = "True binary: only 0 and 1",
        highlights = {{positions = {0,1,2,3,4,5,6,7}, color = "cyan"}}
    }
end
