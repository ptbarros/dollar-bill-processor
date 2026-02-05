--[[
Pattern: SUPER_LOW_SERIAL
DisplayName: Super Low Serial
Description: Serial starting with 6-7 leading zeros (under 100)
Tier: 1
Examples: ["00000001", "00000012", "00000099", "00000005"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Check for 7 leading zeros (single digit: 00000001-00000009)
    if s:match("^0000000[1-9]$") then
        return {
            matched = true,
            message = "Super low serial: single digit",
            highlights = {
                {positions = {0,1,2,3,4,5,6}, color = "gray"},
                {positions = {7}, color = "gold"}
            }
        }
    end

    -- Check for 6 leading zeros (two digits: 00000010-00000099)
    if s:match("^000000[1-9]%d$") then
        return {
            matched = true,
            message = "Super low serial: under 100",
            highlights = {
                {positions = {0,1,2,3,4,5}, color = "gray"},
                {positions = {6, 7}, color = "gold"}
            }
        }
    end

    return {matched = false}
end
