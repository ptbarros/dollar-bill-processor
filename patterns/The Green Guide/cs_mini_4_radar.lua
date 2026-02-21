--[[
Pattern: CS_MINI_4_RADAR
DisplayName: CS-Mini 4 Radar
Description: A CS-Pair inside a CS-20AK grouped: 4-digit palindrome (ABBA) anywhere in the serial. e.g., M 2442xxxx M or M xx4224xx M.
BookRef: CS-1380
Tier: 5
Examples: ["24420000", "00244200", "00002442"]
Odds: 1 in 900,000
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Find a 4-digit palindrome (ABBA) anywhere in the serial
    -- ABBA: d[i] == d[i+3] and d[i+1] == d[i+2] and d[i] ~= d[i+1]
    for i = 1, 5 do  -- positions 1-5 (last ABBA starts at 5, ends at 8)
        local a = d:sub(i, i)
        local b = d:sub(i + 1, i + 1)
        local b2 = d:sub(i + 2, i + 2)
        local a2 = d:sub(i + 3, i + 3)

        if a == a2 and b == b2 and a ~= b then
            local base = i - 1  -- 0-indexed
            return {
                matched = true,
                group_boxes = {
                    {from = base, to = base + 3, color = "orange", thickness = 3}
                },
                connectors = {
                    {from = base, to = base + 3, color = "orange", style = "arc"},
                    {from = base + 1, to = base + 2, color = "coral", style = "arc"}
                },
                message = a .. b .. b .. a .. " mini 4-radar at position " .. i .. " (CS-1380)"
            }
        end
    end

    return {matched = false}
end
