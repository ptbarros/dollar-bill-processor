--[[
Pattern: CS_TRI_RADAR_BOOKEND
DisplayName: CS-Tri Radar Bookend
Description: First three digits mirror the last three in reverse (e.g., 123xx321). The three outer position-pairs are radar-matched: d[1]=d[8], d[2]=d[7], d[3]=d[6]. Not all three can be the same digit.
BookRef: CS-1010
Tier: 7
Examples: ["12300321", "45600654", "12312321"]
Odds: 1 in 720,000
Price: $20-$200
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- d[1]=d[8], d[2]=d[7], d[3]=d[6]
    if d:sub(1, 1) ~= d:sub(8, 8) or
       d:sub(2, 2) ~= d:sub(7, 7) or
       d:sub(3, 3) ~= d:sub(6, 6) then
        return {matched = false}
    end

    local b1 = d:sub(1, 1)
    local b2 = d:sub(2, 2)
    local b3 = d:sub(3, 3)

    -- Not all three the same digit
    if b1 == b2 and b2 == b3 then
        return {matched = false}
    end

    return {
        matched = true,
        highlights = {
            {positions = {0, 7}, color = "orange"},
            {positions = {1, 6}, color = "coral"},
            {positions = {2, 5}, color = "cyan"}
        },
        connectors = {
            {from = 0, to = 7, color = "orange", style = "arc"},
            {from = 1, to = 6, color = "coral",  style = "arc"},
            {from = 2, to = 5, color = "cyan",   style = "arc"}
        },
        message = b1..b2..b3 .. "xx" .. b3..b2..b1 .. " tri-radar bookend (CS-Tri Radar Bookend)"
    }
end
