--[[
Pattern: CS_MINI_5_REPEATER
DisplayName: CS-Mini 5 Repeater
Description: Two CS-20AKs equidistant and separated by one random digit (ABxAB), anywhere. e.g., M 12x12xxx M or M xxx12x12 M.
BookRef: CS-1560
Tier: 7
Examples: ["12312000", "01231200", "00123120"]
Odds: 1 in ~1,000,000
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Find ABxAB: d[i]==d[i+3], d[i+1]==d[i+4], d[i]!=d[i+1]
    for i = 1, 4 do
        local a = d:sub(i, i)
        local b = d:sub(i + 1, i + 1)
        local a2 = d:sub(i + 3, i + 3)
        local b2 = d:sub(i + 4, i + 4)

        if a == a2 and b == b2 and a ~= b then
            local base = i - 1  -- 0-indexed
            local sep = base + 2
            return {
                matched = true,
                highlights = {
                    {positions = {base, base + 1}, color = "orange"},
                    {positions = {base + 3, base + 4}, color = "orange"},
                    {positions = {sep}, color = "charcoal", style = "x"}
                },
                connectors = {
                    {from = base, to = base + 3, color = "orange", style = "arc"},
                    {from = base + 1, to = base + 4, color = "coral", style = "arc"}
                },
                message = a .. b .. "x" .. a .. b .. " mini 5-repeater at position " .. i .. " (CS-1560)"
            }
        end
    end

    return {matched = false}
end
