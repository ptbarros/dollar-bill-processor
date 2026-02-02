--[[
Pattern: TRAILING_7
Description: Ends with 7 identical digits
Tier: 2
Examples: ["31111111", "28888888"]
Odds: 1 in 1,000,000
Price: $100-$450
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check if last 7 digits are all the same
    local trailing = digits:sub(2, 8)
    local d = trailing:sub(1, 1)

    for i = 2, 7 do
        if trailing:sub(i, i) ~= d then
            return {matched = false}
        end
    end

    -- And the first digit must be different
    if digits:sub(1, 1) == d then
        return {matched = false}  -- That's a solid, not trailing 7
    end

    return {
        matched = true,
        highlights = {
            highlight({0}, "red", "odd out"),
            highlight({1, 2, 3, 4, 5, 6, 7}, "gold", "trailing 7")
        },
        connectors = {},
        message = "Trailing 7 x " .. d
    }
end
