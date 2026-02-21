--[[
Pattern: CS_MINI_7_REPEATER
DisplayName: CS-Mini 7 Repeater
Description: Three CS-20AKs separated by one digit (ABCxABC): a 3-digit pattern repeats across a gap. e.g., M 123x123x M or M x123x123 M.
BookRef: CS-1580
Tier: 6
Examples: ["12301230", "01230123"]
Odds: 1 in ~200,000
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Find ABCxABC: d[i]==d[i+4], d[i+1]==d[i+5], d[i+2]==d[i+6]
    -- Only positions 1 and 2 can start a 7-char pattern in an 8-char serial
    for i = 1, 2 do
        local a = d:sub(i, i)
        local b = d:sub(i + 1, i + 1)
        local c = d:sub(i + 2, i + 2)
        local a2 = d:sub(i + 4, i + 4)
        local b2 = d:sub(i + 5, i + 5)
        local c2 = d:sub(i + 6, i + 6)

        if a == a2 and b == b2 and c == c2 then
            local sub7 = d:sub(i, i + 6)
            if unique_count(sub7) >= 2 then
                local base = i - 1  -- 0-indexed
                local sep = base + 3
                return {
                    matched = true,
                    highlights = {
                        {positions = {base, base + 1, base + 2}, color = "orange"},
                        {positions = {base + 4, base + 5, base + 6}, color = "orange"},
                        {positions = {sep}, color = "gray"}
                    },
                    connectors = {
                        {from = base, to = base + 4, color = "orange", style = "arc"},
                        {from = base + 1, to = base + 5, color = "coral", style = "arc"},
                        {from = base + 2, to = base + 6, color = "cyan", style = "arc"}
                    },
                    message = a .. b .. c .. "x" .. a .. b .. c .. " mini 7-repeater at position " .. i .. " (CS-1580)"
                }
            end
        end
    end

    return {matched = false}
end
