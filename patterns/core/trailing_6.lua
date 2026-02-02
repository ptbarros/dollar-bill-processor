--[[
Pattern: TRAILING_6
Description: Ends with 6 identical digits
Tier: 3
Examples: ["24888888", "12000000"]
Odds: 1 in 100,000
Price: $30-$150
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check if last 6 digits are all the same
    local trailing = digits:sub(3, 8)
    local d = trailing:sub(1, 1)

    for i = 2, 6 do
        if trailing:sub(i, i) ~= d then
            return {matched = false}
        end
    end

    return {
        matched = true,
        highlights = {
            highlight({2, 3, 4, 5, 6, 7}, "gold", "trailing 6")
        },
        connectors = {},
        message = "Trailing 6 x " .. d
    }
end
