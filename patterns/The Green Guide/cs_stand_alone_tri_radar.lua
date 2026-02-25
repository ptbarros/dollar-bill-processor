--[[
Pattern: CS_STAND_ALONE_TRI_RADAR
DisplayName: CS-Stand Alone Tri Radar
Description: A CS-Triple surrounded by zeros. This is also a CS-Mini 5 Radar surrounded by zeros. ABBBA structure where the middle three digits are identical. e.g., M 00122210 M.
BookRef: CS-1760
Tier: 3
Examples: ["00122210", "01222100", "01333100"]
Odds: 1 in 27
Price: $0-$5
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    if d:sub(1, 1) ~= "0" or d:sub(8, 8) ~= "0" then
        return {matched = false}
    end

    -- Find ABBBA (5-char palindrome with triple core) at positions 2-6, 3-7 (1-indexed)
    -- d[i]==d[i+4] (outer pair), d[i+1]==d[i+2]==d[i+3] (triple core), outer != inner
    for i = 2, 4 do
        if i + 4 <= 8 then
            local a  = d:sub(i, i)
            local b1 = d:sub(i + 1, i + 1)
            local b2 = d:sub(i + 2, i + 2)
            local b3 = d:sub(i + 3, i + 3)
            local a2 = d:sub(i + 4, i + 4)

            if a == a2 and b1 == b2 and b2 == b3 and a ~= b1 and a ~= "0" and b1 ~= "0" then
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
                        message = a .. b1 .. b1 .. b1 .. a .. " stand-alone tri-radar at position " .. i .. " (CS-Stand Alone Tri Radar)"
                    }
                end
            end
        end
    end

    return {matched = false}
end
