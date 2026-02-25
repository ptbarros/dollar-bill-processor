--[[
Pattern: CS_STAND_ALONE_MINI_5_RADAR
DisplayName: CS-Stand Alone Mini 5 Radar
Description: A CS-Mini 5 Radar (ABCBA palindrome) surrounded by zeros. A CS-2OAK and CS-3OAK alternating, grouped. e.g., M 00121210 M.
BookRef: CS-1750
Tier: 3
Examples: ["00121210", "01212100", "01232100"]
Odds: 1 in 27
Price: $0-$5
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    if d:sub(1, 1) ~= "0" or d:sub(8, 8) ~= "0" then
        return {matched = false}
    end

    -- Find ABCBA (5-char palindrome) at positions 2-6, 3-7 (1-indexed)
    -- d[i]==d[i+4], d[i+1]==d[i+3], all non-zero, all others zero
    for i = 2, 4 do
        if i + 4 <= 8 then
            local a  = d:sub(i, i)
            local b  = d:sub(i + 1, i + 1)
            local c  = d:sub(i + 2, i + 2)
            local b2 = d:sub(i + 3, i + 3)
            local a2 = d:sub(i + 4, i + 4)

            if a == a2 and b == b2 and a ~= "0" and b ~= "0" and c ~= "0" then
                -- All other positions must be zero
                local all_others_zero = true
                for j = 1, 8 do
                    if j < i or j > i + 4 then
                        if d:sub(j, j) ~= "0" then
                            all_others_zero = false
                            break
                        end
                    end
                end
                if all_others_zero then
                    local base = i - 1  -- 0-indexed
                    return {
                        matched = true,
                        group_boxes = {
                            {from = base, to = base + 4, color = "orange", thickness = 3}
                        },
                        connectors = {
                            {from = base, to = base + 4, color = "orange", style = "arc"},
                            {from = base + 1, to = base + 3, color = "coral", style = "arc"}
                        },
                        message = a .. b .. c .. b .. a .. " stand-alone mini 5-radar at position " .. i .. " (CS-Stand Alone Mini 5 Radar)"
                    }
                end
            end
        end
    end

    return {matched = false}
end
