--[[
Pattern: NICKS_TRUE_TRINARY
DisplayName: True Trinary
Description: Only digits 0, 1, 2 used, all three present
Tier: 4
Examples: ["01201201", "00112200", "12012012", "22110022"]
--]]

function match(ctx)
    local s = ctx.digits

    local has_0 = false
    local has_1 = false
    local has_2 = false

    for i = 1, 8 do
        local d = s:sub(i, i)
        if d == "0" then
            has_0 = true
        elseif d == "1" then
            has_1 = true
        elseif d == "2" then
            has_2 = true
        else
            return {matched = false}
        end
    end

    if has_0 and has_1 and has_2 then
        return {
            matched = true,
            message = "True trinary: only 0, 1, 2",
            highlights = {{positions = {0,1,2,3,4,5,6,7}, color = "lime"}}
        }
    end

    return {matched = false}
end
