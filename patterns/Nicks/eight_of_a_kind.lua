--[[
Pattern: EIGHT_OF_A_KIND
DisplayName: 8 of a Kind
Description: All 8 digits are the same (solid)
Tier: 1
Examples: ["00000000", "11111111", "22222222", "55555555", "99999999"]
--]]

function match(ctx)
    local s = ctx.digits
    local first = s:sub(1, 1)

    for i = 2, 8 do
        if s:sub(i, i) ~= first then
            return {matched = false}
        end
    end

    return {
        matched = true,
        message = "Solid " .. first .. "s - all 8 digits identical",
        highlights = {{positions = {0,1,2,3,4,5,6,7}, color = "gold"}}
    }
end
