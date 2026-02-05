--[[
Pattern: VERY_LOW_SERIAL
DisplayName: Very Low Serial
Description: Serial starting with 4-5 leading zeros (100-9999)
Tier: 2
Examples: ["00000100", "00000999", "00001000", "00009999"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Check for 5 leading zeros (3 digits: 00000100-00000999)
    if s:match("^00000[1-9]%d%d$") then
        return {
            matched = true,
            message = "Very low serial: 3 digits",
            highlights = {
                {positions = {0,1,2,3,4}, color = "gray"},
                {positions = {5,6,7}, color = "gold"}
            }
        }
    end

    -- Check for 4 leading zeros (4 digits: 00001000-00009999)
    if s:match("^0000[1-9]%d%d%d$") then
        return {
            matched = true,
            message = "Very low serial: 4 digits",
            highlights = {
                {positions = {0,1,2,3}, color = "gray"},
                {positions = {4,5,6,7}, color = "gold"}
            }
        }
    end

    return {matched = false}
end
