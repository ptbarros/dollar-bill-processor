--[[
Pattern: TRAILING_5
Description: Ends with 5 identical digits
Tier: 4
Examples: ["12300000", "56711111"]
Odds: 1 in 10,000
Price: $10-$50
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check if last 5 digits are all the same
    local trailing = digits:sub(4, 8)
    local d = trailing:sub(1, 1)

    for i = 2, 5 do
        if trailing:sub(i, i) ~= d then
            return {matched = false}
        end
    end

    return {
        matched = true,
        highlights = {
            highlight({3, 4, 5, 6, 7}, "gold", "trailing 5")
        },
        connectors = {},
        message = "Trailing 5 x " .. d
    }
end
