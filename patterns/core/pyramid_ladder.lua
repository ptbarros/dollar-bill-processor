--[[
Pattern: PYRAMID_LADDER
Description: Up then down (ABCDEDCBA or BCDEDCBA)
Tier: 2
Examples: ["12321000", "23432100", "34543212"]
Odds: 1 in 1,200,000
Price: $10-$50
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Try to find a 5-digit pyramid (12321) anywhere in the 8 digits
    for start = 0, 3 do
        local sub = digits:sub(start + 1, start + 5)

        -- Check ascending: positions 1,2 go up
        local d1 = tonumber(sub:sub(1, 1))
        local d2 = tonumber(sub:sub(2, 2))
        local d3 = tonumber(sub:sub(3, 3))  -- peak
        local d4 = tonumber(sub:sub(4, 4))
        local d5 = tonumber(sub:sub(5, 5))

        -- Pattern: d1 < d2 < d3 > d4 > d5, each step is 1
        if d2 == d1 + 1 and d3 == d2 + 1 and d4 == d3 - 1 and d5 == d4 - 1 then
            -- Found a 5-digit pyramid
            local p0 = start
            local p1 = start + 1
            local p2 = start + 2  -- peak
            local p3 = start + 3
            local p4 = start + 4

            return {
                matched = true,
                highlights = {
                    {positions = {p0, p1}, color = "lime"},
                    {positions = {p2}, color = "yellow"},
                    {positions = {p3, p4}, color = "cyan"}
                },
                connectors = {
                    {from = p0, to = p4, color = "orange", style = "arc"},
                    {from = p1, to = p3, color = "orange", style = "arc"}
                },
                message = "Pyramid: " .. sub
            }
        end
    end

    -- Try to find a 7-digit pyramid (1234321) anywhere in the 8 digits
    for start = 0, 1 do
        local sub = digits:sub(start + 1, start + 7)

        local d1 = tonumber(sub:sub(1, 1))
        local d2 = tonumber(sub:sub(2, 2))
        local d3 = tonumber(sub:sub(3, 3))
        local d4 = tonumber(sub:sub(4, 4))  -- peak
        local d5 = tonumber(sub:sub(5, 5))
        local d6 = tonumber(sub:sub(6, 6))
        local d7 = tonumber(sub:sub(7, 7))

        -- Check: d1 < d2 < d3 < d4 > d5 > d6 > d7, each step is 1
        if d2 == d1 + 1 and d3 == d2 + 1 and d4 == d3 + 1 and
           d5 == d4 - 1 and d6 == d5 - 1 and d7 == d6 - 1 then
            -- Found a 7-digit pyramid
            local p0 = start
            local p1 = start + 1
            local p2 = start + 2
            local p3 = start + 3  -- peak
            local p4 = start + 4
            local p5 = start + 5
            local p6 = start + 6

            return {
                matched = true,
                highlights = {
                    {positions = {p0, p1, p2}, color = "lime"},
                    {positions = {p3}, color = "yellow"},
                    {positions = {p4, p5, p6}, color = "cyan"}
                },
                connectors = {
                    {from = p0, to = p6, color = "orange", style = "arc"},
                    {from = p1, to = p5, color = "orange", style = "arc"},
                    {from = p2, to = p4, color = "orange", style = "arc"}
                },
                message = "Pyramid: " .. sub
            }
        end
    end

    return {matched = false}
end
