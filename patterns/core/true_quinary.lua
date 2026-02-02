--[[
Pattern: TRUE_QUINARY
Description: Only contains digits 0, 1, 2, 3, 4
Tier: 4
Examples: ["01234123", "12340123", "00112233"]
Odds: 1 in 245
Price: $3-$10
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    if not only_digits(digits, "01234") then
        return {matched = false}
    end

    -- Highlight all positions
    local highlights = {}
    for i = 0, 7 do
        local d = digits:sub(i + 1, i + 1)
        local colors = {["0"] = "blue", ["1"] = "cyan", ["2"] = "teal", ["3"] = "lime", ["4"] = "gold"}
        table.insert(highlights, highlight({i}, colors[d], d))
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = {},
        message = "True quinary (0-4 only)"
    }
end
