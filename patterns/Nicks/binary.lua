--[[
Pattern: NICKS_BINARY
DisplayName: Binary
Description: Exactly 2 unique digits in the serial
Tier: 5
Examples: ["12121212", "00110011", "99889988", "55665566"]
--]]

function match(ctx)
    local s = ctx.digits

    local unique = {}
    for i = 1, 8 do
        unique[s:sub(i, i)] = true
    end

    local count = 0
    for _ in pairs(unique) do count = count + 1 end

    if count == 2 then
        return {
            matched = true,
            message = "Binary: 2 unique digits",
            highlights = {{positions = {0,1,2,3,4,5,6,7}, color = "cyan"}}
        }
    end

    return {matched = false}
end
