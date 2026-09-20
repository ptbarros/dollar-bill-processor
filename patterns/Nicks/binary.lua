--[[
Pattern: NICKS_BINARY
DisplayName: Binary
Description: Exactly 2 unique digits in the serial
Tier: 5
Examples: ["12121212", "00110011", "99889988", "55665566"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Two distinct digits, captured in first-appearance order for stable colours.
    local order = {}
    local seen = {}
    for i = 1, 8 do
        local c = s:sub(i, i)
        if not seen[c] then
            seen[c] = true
            order[#order + 1] = c
        end
    end

    if #order ~= 2 then return {matched = false} end

    -- A coloured box per digit: each of the two digits gets its own colour, so the
    -- two groups read apart at a glance (Ed review / Workshop draft).
    local colors = {"blue", "orange"}
    local color_of = {}
    for k, c in ipairs(order) do color_of[c] = colors[k] end

    local highlights = {}
    for i = 1, 8 do
        local c = s:sub(i, i)
        highlights[#highlights + 1] = {positions = {i - 1}, color = color_of[c], style = "box"}
    end

    return {
        matched = true,
        message = "Binary: 2 unique digits",
        highlights = highlights
    }
end
