--[[
Pattern: SUM_62_OR_10
Description: Digit sum equals 62 or 10
Tier: 5
Examples: ["00008020", "90100000", "21101005"]
Odds: 1 in 2,272
Price: $5-$25+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local sum = digit_sum(digits)
    if sum ~= 62 and sum ~= 10 then
        return {matched = false}
    end

    -- Skip zeros: no box on any 0 digit (Ed review, applies to all Sum patterns).
    local positions = {}
    for i = 0, 7 do
        if digits:sub(i + 1, i + 1) ~= "0" then
            table.insert(positions, i)
        end
    end

    return {
        matched = true,
        highlights = {
            highlight(positions, "purple", "sum")
        },
        connectors = {},
        message = "Digit sum = " .. sum
    }
end
