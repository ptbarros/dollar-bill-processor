--[[
Pattern: CS_MINI_4_REPEATER
DisplayName: CS-Mini 4 Repeater
Description: Two different digits repeated and grouped together (ABAB), anywhere in the serial. e.g., M x2121xxx M or M xxxx2121 M.
BookRef: CS-1550
Tier: 8
Examples: ["21210000", "02121000", "00212100"]
Odds: 1 in ~2,000,000
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Find ABAB anywhere: d[i]==d[i+2], d[i+1]==d[i+3], d[i]!=d[i+1]
    for i = 1, 5 do
        local a = d:sub(i, i)
        local b = d:sub(i + 1, i + 1)
        local a2 = d:sub(i + 2, i + 2)
        local b2 = d:sub(i + 3, i + 3)

        if a == a2 and b == b2 and a ~= b then
            local base = i - 1  -- 0-indexed
            return {
                matched = true,
                group_boxes = {
                    {from = base, to = base + 3, color = "teal", thickness = 2}
                },
                highlights = {
                    {positions = {base, base + 2}, color = "orange"},
                    {positions = {base + 1, base + 3}, color = "coral"}
                },
                message = a .. b .. a .. b .. " mini 4-repeater at position " .. i .. " (CS-1550)"
            }
        end
    end

    return {matched = false}
end
