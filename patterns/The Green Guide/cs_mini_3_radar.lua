--[[
Pattern: CS_MINI_3_RADAR
DisplayName: CS-Mini 3 Radar
Description: A CS-20AK separated by one non-similar digit: 3-digit palindrome (AXA) anywhere. Also the CS-Mini 3 Repeater. e.g., M xx1x1xxx M.
BookRef: CS-1340
Tier: 8
Examples: ["12132456", "11213456", "12345616"]
Odds: 1 in ~100,000
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Find AXA anywhere: d[i] == d[i+2] and d[i] ~= d[i+1]
    for i = 1, 6 do
        local a = d:sub(i, i)
        local x = d:sub(i + 1, i + 1)
        local a2 = d:sub(i + 2, i + 2)

        if a == a2 and a ~= x then
            local base = i - 1  -- 0-indexed
            return {
                matched = true,
                highlights = {
                    {positions = {base, base + 2}, color = "orange"},
                    {positions = {base + 1}, color = "gray"}
                },
                connectors = {
                    {from = base, to = base + 2, color = "orange", style = "arc"}
                },
                message = a .. x .. a .. " mini 3-radar at position " .. i .. " (CS-1340)"
            }
        end
    end

    return {matched = false}
end
