--[[
Pattern: CS_MINI_7_RADAR
DisplayName: CS-Mini 7 Radar
Description: Three equidistant CS-20AKs separated by one digit: 7-digit palindrome (ABCDCBA) anywhere in the serial. e.g., M x234x432 M or M 234x432x M.
BookRef: CS-1410
Tier: 3
Examples: ["02341432", "12341432", "23414320"]
Odds: 1 in 999,999
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Find a 7-digit palindrome (ABCDCBA): d[i]==d[i+6], d[i+1]==d[i+5], d[i+2]==d[i+4]
    -- Only two positions possible: i=1 (positions 1-7) or i=2 (positions 2-8)
    for i = 1, 2 do
        local a = d:sub(i, i)
        local b = d:sub(i + 1, i + 1)
        local c = d:sub(i + 2, i + 2)
        local c2 = d:sub(i + 4, i + 4)
        local b2 = d:sub(i + 5, i + 5)
        local a2 = d:sub(i + 6, i + 6)

        if a == a2 and b == b2 and c == c2 then
            local sub = d:sub(i, i + 6)
            if unique_count(sub) >= 2 then
                local base = i - 1  -- 0-indexed
                local mid = base + 3
                return {
                    matched = true,
                    highlights = {
                        {positions = {base, base + 6}, color = "orange"},
                        {positions = {base + 1, base + 5}, color = "coral"},
                        {positions = {base + 2, base + 4}, color = "cyan"},
                        {positions = {mid}, color = "charcoal", style = "x"}
                    },
                    connectors = {
                        {from = base, to = base + 6, color = "orange", style = "arc"},
                        {from = base + 1, to = base + 5, color = "coral", style = "arc"},
                        {from = base + 2, to = base + 4, color = "cyan", style = "arc"}
                    },
                    message = a .. b .. c .. "x" .. c .. b .. a .. " mini 7-radar at position " .. i .. " (CS-1410)"
                }
            end
        end
    end

    return {matched = false}
end
