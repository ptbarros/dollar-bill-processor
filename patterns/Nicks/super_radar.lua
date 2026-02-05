--[[
Pattern: NICKS_SUPER_RADAR
DisplayName: Super Radar
Description: First and last digit match, middle 6 are identical (ABBBBBBA)
Tier: 2
Examples: ["12222221", "01111110", "98888889", "10000001"]
--]]

function match(ctx)
    local s = ctx.digits

    -- First and last must match
    if s:sub(1, 1) ~= s:sub(8, 8) then
        return {matched = false}
    end

    -- Middle 6 must all be identical
    local middle_digit = s:sub(2, 2)
    for i = 3, 7 do
        if s:sub(i, i) ~= middle_digit then
            return {matched = false}
        end
    end

    return {
        matched = true,
        message = "Super radar: " .. s:sub(1, 1) .. " + " .. middle_digit .. "×6 + " .. s:sub(8, 8),
        highlights = {
            {positions = {0, 7}, color = "gold"},
            {positions = {1, 2, 3, 4, 5, 6}, color = "orange"}
        },
        connectors = {{from = 0, to = 7, color = "gold", style = "arc"}}
    }
end
