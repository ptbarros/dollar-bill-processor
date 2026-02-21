--[[
Pattern: CS_STAND_ALONE_PAIR
DisplayName: CS-Stand Alone Pair
Description: A CS-Pair (two consecutive identical digits) surrounded by zeros. e.g., M 00220000 M or M 00000220 M.
BookRef: CS-1660
Tier: 5
Examples: ["00220000", "00002200", "02200000"]
Odds: 1 in 45
Price: $25-$500
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    if d:sub(1, 1) ~= "0" or d:sub(8, 8) ~= "0" then
        return {matched = false}
    end

    -- Find a pair of consecutive identical non-zero digits; all other positions must be zero
    for i = 2, 7 do  -- pair can start at positions 2-7 (must fit within 2-7)
        if i + 1 <= 7 then
            local a = d:sub(i, i)
            local b = d:sub(i + 1, i + 1)
            if a == b and a ~= "0" then
                -- All other positions must be zero
                local all_others_zero = true
                for j = 1, 8 do
                    if j ~= i and j ~= i + 1 then
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
                            {from = base, to = base + 1, color = "gold", thickness = 3}
                        },
                        message = a .. a .. " stand-alone pair at position " .. i .. " (CS-Stand Alone Pair)"
                    }
                end
            end
        end
    end

    return {matched = false}
end
