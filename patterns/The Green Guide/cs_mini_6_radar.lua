--[[
Pattern: CS_MINI_6_RADAR
DisplayName: CS-Mini 6 Radar
Description: One CS-Pair surrounded by two equidistant CS-20AKs: 6-digit palindrome (ABCCBA) anywhere in the serial. e.g., M 234432xx M or M x221122x M or M xx234432 M.
BookRef: CS-1400
Tier: 4
Examples: ["23443200", "02344320", "00234432"]
Odds: 1 in 99,000
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Find a 6-digit palindrome (ABCCBA) anywhere: d[i]==d[i+5], d[i+1]==d[i+4], d[i+2]==d[i+3]
    for i = 1, 3 do  -- positions 1-3 (last 6-palindrome starts at 3, ends at 8)
        local a = d:sub(i, i)
        local b = d:sub(i + 1, i + 1)
        local c = d:sub(i + 2, i + 2)
        local c2 = d:sub(i + 3, i + 3)
        local b2 = d:sub(i + 4, i + 4)
        local a2 = d:sub(i + 5, i + 5)

        if a == a2 and b == b2 and c == c2 then
            -- Must have at least two distinct digits to avoid trivial palindromes
            local sub = d:sub(i, i + 5)
            if unique_count(sub) >= 2 then
                local base = i - 1  -- 0-indexed
                return {
                    matched = true,
                    group_boxes = {
                        {from = base, to = base + 5, color = "orange", thickness = 3}
                    },
                    connectors = {
                        {from = base, to = base + 5, color = "orange", style = "arc"},
                        {from = base + 1, to = base + 4, color = "coral", style = "arc"},
                        {from = base + 2, to = base + 3, color = "cyan", style = "arc"}
                    },
                    message = a .. b .. c .. c .. b .. a .. " mini 6-radar at position " .. i .. " (CS-1400)"
                }
            end
        end
    end

    return {matched = false}
end
