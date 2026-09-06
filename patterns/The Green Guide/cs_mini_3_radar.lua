--[[
Pattern: CS_MINI_3_RADAR
DisplayName: CS-Mini 3 Radar
Description: A mirrored 3-digit window (AXA) anywhere in the serial — the same digit at positions i and i+2 with a different digit sandwiched between them. The window can begin at any of the first six positions.
BookRef: CS-1370
Tier: 8
Examples: ["34356789", "45056789", "23456787"]
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
                    {positions = {base + 1}, color = "charcoal", style = "x"}
                },
                connectors = {
                    {from = base, to = base + 2, color = "orange", style = "arc"}
                },
                message = a .. x .. a .. " mini 3-radar at position " .. i .. " (CS-1370)"
            }
        end
    end

    return {matched = false}
end
