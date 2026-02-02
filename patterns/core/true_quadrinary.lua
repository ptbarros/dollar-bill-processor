--[[
Pattern: TRUE_QUADRINARY
Description: Only contains digits 0, 1, 2, 3
Tier: 4
Examples: ["01230123", "12301230", "00112233"]
Odds: 1 in 1,465
Price: $5-$25
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    if not only_digits(digits, "0123") then
        return {matched = false}
    end

    -- Highlight all positions in a gradient
    local highlights = {}
    for i = 0, 7 do
        local d = digits:sub(i + 1, i + 1)
        local colors = {["0"] = "blue", ["1"] = "cyan", ["2"] = "teal", ["3"] = "lime"}
        table.insert(highlights, highlight({i}, colors[d], d))
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = {},
        message = "True quadrinary (0-3 only)"
    }
end
