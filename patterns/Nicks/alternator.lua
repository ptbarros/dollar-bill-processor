--[[
Pattern: NICKS_ALTERNATOR
DisplayName: Alternator
Description: Odd positions all same OR even positions all same (X_X_X_X_ or _Y_Y_Y_Y)
Tier: 5
Examples: ["12131415", "10203040", "91929394", "19293949"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Check odd positions (1,3,5,7)
    local odd_digit = s:sub(1, 1)
    local odd_same = true
    for i = 3, 7, 2 do
        if s:sub(i, i) ~= odd_digit then
            odd_same = false
            break
        end
    end

    if odd_same then
        return {
            matched = true,
            message = "Alternator: " .. odd_digit .. " in odd positions",
            highlights = {
                {positions = {0, 2, 4, 6}, color = "orange"},
                {positions = {1, 3, 5, 7}, color = "gray"}
            }
        }
    end

    -- Check even positions (2,4,6,8)
    local even_digit = s:sub(2, 2)
    local even_same = true
    for i = 4, 8, 2 do
        if s:sub(i, i) ~= even_digit then
            even_same = false
            break
        end
    end

    if even_same then
        return {
            matched = true,
            message = "Alternator: " .. even_digit .. " in even positions",
            highlights = {
                {positions = {1, 3, 5, 7}, color = "orange"},
                {positions = {0, 2, 4, 6}, color = "gray"}
            }
        }
    end

    return {matched = false}
end
