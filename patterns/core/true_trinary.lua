--[[
Pattern: NICKS_TRUE_TRINARY
DisplayName: True Trinary
Description: Only digits 0, 1, 2 used, all three present
Tier: 5
Odds: 1 in 16,563 (5,796 per 96M)
Examples: ["01201201", "00112200", "12012012", "22110022"]
--]]

function match(ctx)
    local s = ctx.digits

    local has_0 = false
    local has_1 = false
    local has_2 = false

    for i = 1, 8 do
        local d = s:sub(i, i)
        if d == "0" then
            has_0 = true
        elseif d == "1" then
            has_1 = true
        elseif d == "2" then
            has_2 = true
        else
            return {matched = false}
        end
    end

    if has_0 and has_1 and has_2 then
        -- One colored box per distinct digit value (Ed review).
        local colors = {"blue", "orange", "magenta", "red"}
        local by_digit, order = {}, {}
        for i = 0, 7 do
            local ch = s:sub(i + 1, i + 1)
            if not by_digit[ch] then by_digit[ch] = {}; table.insert(order, ch) end
            table.insert(by_digit[ch], i)
        end
        local hl = {}
        for idx, ch in ipairs(order) do
            table.insert(hl, {positions = by_digit[ch], color = colors[((idx - 1) % #colors) + 1]})
        end
        return {
            matched = true,
            message = "True trinary: only 0, 1, 2",
            highlights = hl
        }
    end

    return {matched = false}
end
