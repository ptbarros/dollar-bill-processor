--[[
Pattern: MULTI_MILLIONAIRE
Description: Multi-millionaire (X0000000)
Tier: 1
Examples: ["10000000", "50000000", "70000000"]
Odds: 1 in 10,666,667
Price: $100-$9,000
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- First digit must be 1-9
    local first = digits:sub(1, 1)
    if first == "0" then
        return {matched = false}
    end

    -- Rest must be zeros
    if not ends_with(digits, "0000000") then
        return {matched = false}
    end

    local millions = tonumber(first) * 10
    return {
        matched = true,
        highlights = {
            highlight({0}, "yellow", "tens of millions"),
            highlight({1, 2, 3, 4, 5, 6, 7}, "gold", "zeros")
        },
        connectors = {},
        message = millions .. " Million"
    }
end
