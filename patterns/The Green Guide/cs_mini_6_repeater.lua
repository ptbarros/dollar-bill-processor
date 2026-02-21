--[[
Pattern: CS_MINI_6_REPEATER
DisplayName: CS-Mini 6 Repeater
Description: Any 3-digit pattern repeated twice and grouped (ABCABC), anywhere in the serial. e.g., M 301301xx M or M xx301301 M.
BookRef: CS-1570
Tier: 7
Examples: ["30130100", "03013010", "00301301"]
Odds: 1 in ~100,000
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Find ABCABC: d[i]==d[i+3], d[i+1]==d[i+4], d[i+2]==d[i+5]
    -- At least 2 distinct digits in the 6-char block
    for i = 1, 3 do
        local a = d:sub(i, i)
        local b = d:sub(i + 1, i + 1)
        local c = d:sub(i + 2, i + 2)
        local a2 = d:sub(i + 3, i + 3)
        local b2 = d:sub(i + 4, i + 4)
        local c2 = d:sub(i + 5, i + 5)

        if a == a2 and b == b2 and c == c2 then
            local sub6 = d:sub(i, i + 5)
            if unique_count(sub6) >= 2 then
                local base = i - 1  -- 0-indexed
                return {
                    matched = true,
                    group_boxes = {
                        {from = base, to = base + 2, color = "orange", thickness = 2},
                        {from = base + 3, to = base + 5, color = "orange", thickness = 2}
                    },
                    connectors = {
                        {from = base, to = base + 3, color = "orange", style = "arc"},
                        {from = base + 1, to = base + 4, color = "coral", style = "arc"},
                        {from = base + 2, to = base + 5, color = "cyan", style = "arc"}
                    },
                    message = a .. b .. c .. " repeats at position " .. i .. " (CS-Mini 6 Repeater)"
                }
            end
        end
    end

    return {matched = false}
end
