--[[
Pattern: TRAILING_QUAD
Description: Ends with 4 identical digits
Tier: 4
Examples: ["12340000", "93891111", "56781111"]
Odds: 1 in 1,000
Price: $3-$20
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check if last 4 digits are all the same
    local trailing = digits:sub(5, 8)
    local d = trailing:sub(1, 1)

    for i = 2, 4 do
        if trailing:sub(i, i) ~= d then
            return {matched = false}
        end
    end

    return {
        matched = true,
        highlights = {
            highlight({4, 5, 6, 7}, "gold", "trailing quad")
        },
        connectors = {},
        message = "Trailing quad " .. d .. "s"
    }
end
