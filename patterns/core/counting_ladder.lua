--[[
Pattern: COUNTING_LADDER
Description: Counting pattern (10203040)
Tier: 7
Examples: ["10203040", "20304050"]
Odds: 1 in 558,139
Price: $30-$200
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check X0Y0Z0W0 pattern where X,Y,Z,W are consecutive
    if digits:sub(2,2) ~= "0" or digits:sub(4,4) ~= "0" or
       digits:sub(6,6) ~= "0" or digits:sub(8,8) ~= "0" then
        return {matched = false}
    end

    local d1 = tonumber(digits:sub(1,1))
    local d2 = tonumber(digits:sub(3,3))
    local d3 = tonumber(digits:sub(5,5))
    local d4 = tonumber(digits:sub(7,7))

    -- Check ascending sequence
    if d2 == d1 + 1 and d3 == d2 + 1 and d4 == d3 + 1 then
        return {
            matched = true,
            highlights = {
                highlight({0}, "lime", "first"),
                highlight({2}, "teal", "second"),
                highlight({4}, "cyan", "third"),
                highlight({6}, "blue", "fourth"),
                highlight({1, 3, 5, 7}, "gray", "zeros")
            },
            connectors = {
                connector(0, 2, "lime", "line"),
                connector(2, 4, "lime", "line"),
                connector(4, 6, "lime", "line")
            },
            message = string.format("Counting ladder: %d0%d0%d0%d0", d1, d2, d3, d4)
        }
    end

    return {matched = false}
end
