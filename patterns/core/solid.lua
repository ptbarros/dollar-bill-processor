--[[
Pattern: SOLID
Description: All 8 digits are identical (e.g., 88888888)
Tier: 1
Examples: ["88888888", "11111111", "00000000"]
Odds: 1 in 10,000,000
Price: $1,000-$10,000+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check if all digits are the same
    local first = digits:sub(1, 1)
    for i = 2, 8 do
        if digits:sub(i, i) ~= first then
            return {matched = false}
        end
    end

    -- All positions highlighted in yellow
    local highlights = {}
    for i = 0, 7 do
        table.insert(highlights, {positions = {i}, color = "yellow", label = "solid"})
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = {},
        message = "Perfect solid - all " .. first .. "s"
    }
end
