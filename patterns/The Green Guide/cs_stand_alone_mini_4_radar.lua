--[[
Pattern: CS_STAND_ALONE_MINI_4_RADAR
DisplayName: CS-Stand Alone Mini 4 Radar
Description: A CS-Mini 4 Radar (ABBA palindrome) surrounded by zeros. e.g., M 01221000 M or M 00122100 M.
BookRef: CS-1740
Tier: 3
Examples: ["01221000", "00122100", "00012210"]
Odds: 1 in 27
Price: $0-$5
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    if d:sub(1, 1) ~= "0" or d:sub(8, 8) ~= "0" then
        return {matched = false}
    end

    -- Find ABBA (4-char palindrome) at positions 2-5, 3-6, or 4-7
    -- With all other positions being zero
    for i = 2, 5 do
        if i + 3 <= 7 then  -- ABBA must end at or before position 7
            local a = d:sub(i, i)
            local b = d:sub(i + 1, i + 1)
            local b2 = d:sub(i + 2, i + 2)
            local a2 = d:sub(i + 3, i + 3)

            if a == a2 and b == b2 and a ~= b and a ~= "0" then
                -- All other positions must be zero
                local all_others_zero = true
                for j = 1, 8 do
                    if j < i or j > i + 3 then
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
                            {from = base, to = base + 3, color = "orange", thickness = 3}
                        },
                        connectors = {
                            {from = base, to = base + 3, color = "orange", style = "arc"},
                            {from = base + 1, to = base + 2, color = "coral", style = "arc"}
                        },
                        message = a .. b .. b .. a .. " stand-alone mini 4-radar at position " .. i .. " (CS-Stand Alone Mini 4 Radar)"
                    }
                end
            end
        end
    end

    return {matched = false}
end
