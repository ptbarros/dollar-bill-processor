--[[
Pattern: NICKS_TRINARY
DisplayName: Trinary
Description: Exactly 3 unique digits in the serial
Tier: 6
Examples: ["12121212", "00112211", "12312312", "99887799"]
--]]

function match(ctx)
    local s = ctx.digits

    local unique = {}
    for i = 1, 8 do
        unique[s:sub(i, i)] = true
    end

    local count = 0
    for _ in pairs(unique) do count = count + 1 end

    if count == 3 then
        return {
            matched = true,
            message = "Trinary: 3 unique digits",
            highlights = {{positions = {0,1,2,3,4,5,6,7}, color = "purple"}}
        }
    end

    return {matched = false}
end
