--[[
Pattern: CS_STAND_ALONE_DOUBLE_REPEATER
DisplayName: CS-Stand Alone Double Repeater
Description: Two CS-20AKs that repeat twice within zeros (ABAB surrounded by zeros). e.g., M 01212000 M or M 00121200 M.
BookRef: CS-1710
Tier: 4
Examples: ["01212000", "00121200", "00012120"]
Odds: 1 in 396
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    if d:sub(1, 1) ~= "0" or d:sub(8, 8) ~= "0" then
        return {matched = false}
    end

    -- Find ABAB (4-char) at positions 2-5, 3-6, or 4-7 (all within the zero-bounded range)
    -- With all other positions being zero
    for i = 2, 5 do
        if i + 3 <= 7 then  -- ABAB must end at or before position 7
            local a = d:sub(i, i)
            local b = d:sub(i + 1, i + 1)
            local a2 = d:sub(i + 2, i + 2)
            local b2 = d:sub(i + 3, i + 3)

            if a == a2 and b == b2 and a ~= b and a ~= "0" and b ~= "0" then
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
                        highlights = {
                            {positions = {base, base + 2}, color = "orange"},
                            {positions = {base + 1, base + 3}, color = "coral"}
                        },
                        message = a .. b .. a .. b .. " stand-alone repeater at position " .. i .. " (CS-Stand Alone Double Repeater)"
                    }
                end
            end
        end
    end

    return {matched = false}
end
