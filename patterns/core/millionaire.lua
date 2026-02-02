--[[
Pattern: MILLIONAIRE
Description: Millionaire note (XX000000 or 0X000000)
Tier: 2
Examples: ["12000000", "10000000", "01000000"]
Odds: 1 in 1,103,448
Price: $80-$1,000
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Last 6 must be zeros
    if not ends_with(digits, "000000") then
        return {matched = false}
    end

    -- First 2 digits determine the millionaire value
    local millions = tonumber(digits:sub(1, 2))
    if millions == 0 then
        return {matched = false}  -- That would be under 1 million
    end

    return {
        matched = true,
        highlights = {
            highlight({0, 1}, "yellow", "millions"),
            highlight({2, 3, 4, 5, 6, 7}, "gold", "zeros")
        },
        connectors = {},
        message = millions .. " Million"
    }
end
