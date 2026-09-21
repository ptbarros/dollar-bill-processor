--[[
Pattern: SUM_1
Description: Sum equals 1
Tier: 1
Examples: ["10000000", "00001000", "00000001"]
Odds: 1 in 12,000,000 (8 per 96M)
Price: $100-$1,500
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local sum = digit_sum(digits)
    if sum ~= 1 then
        return {matched = false}
    end

    -- Find the single 1 digit
    local one_pos = nil
    for i = 0, 7 do
        if digits:sub(i + 1, i + 1) == "1" then
            one_pos = i
            break
        end
    end

    -- Box only the single 1; skip the zeros (Ed review).
    return {
        matched = true,
        highlights = {
            highlight({one_pos}, "gold", "sum=1")
        },
        connectors = {},
        message = "Digit sum = 1"
    }
end
