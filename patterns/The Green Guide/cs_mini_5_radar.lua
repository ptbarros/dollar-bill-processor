--[[
Pattern: CS_MINI_5_RADAR
DisplayName: CS-Mini 5 Radar
Description: A CS-20AK separated by one digit with another CS-20AK outside: 5-digit palindrome (ABCBA) anywhere in the serial. e.g., M 24x42xxx M or M xxx24x42 M.
BookRef: CS-1390
Tier: 5
Examples: ["24342000", "00024342", "02434200"]
Odds: 1 in 729,000
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Find a 5-digit palindrome (ABCBA) anywhere: d[i]==d[i+4], d[i+1]==d[i+3]
    -- A ~= B (otherwise it's a different structure)
    for i = 1, 4 do  -- positions 1-4 (last ABCBA starts at 4, ends at 8)
        local a = d:sub(i, i)
        local b = d:sub(i + 1, i + 1)
        local b2 = d:sub(i + 3, i + 3)
        local a2 = d:sub(i + 4, i + 4)

        if a == a2 and b == b2 and a ~= b then
            local base = i - 1  -- 0-indexed
            local mid = base + 2
            return {
                matched = true,
                highlights = {
                    {positions = {base, base + 4}, color = "orange"},
                    {positions = {base + 1, base + 3}, color = "coral"},
                    {positions = {mid}, color = "gray"}
                },
                connectors = {
                    {from = base, to = base + 4, color = "orange", style = "arc"},
                    {from = base + 1, to = base + 3, color = "coral", style = "arc"}
                },
                message = a .. b .. "x" .. b .. a .. " mini 5-radar at position " .. i .. " (CS-1390)"
            }
        end
    end

    return {matched = false}
end
