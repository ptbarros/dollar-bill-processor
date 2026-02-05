--[[
Pattern: LOW_SERIAL
DisplayName: Low Serial
Description: Serial starting with 3 leading zeros (10000-99999)
Tier: 3
Examples: ["00010000", "00012345", "00099999", "00050000"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Check for 3 leading zeros (5 digits: 00010000-00099999)
    if s:match("^000[1-9]%d%d%d%d$") then
        return {
            matched = true,
            message = "Low serial: 5 digits",
            highlights = {
                {positions = {0,1,2}, color = "gray"},
                {positions = {3,4,5,6,7}, color = "gold"}
            }
        }
    end

    return {matched = false}
end
