--[[
Pattern: CS_STAND_ALONE_QUINT
DisplayName: CS-Stand Alone Quint
Description: A CS-Quint (5 consecutive identical non-zero digits) surrounded by zeros. e.g., M 00555550 M or M 05555500 M.
BookRef: CS-1690
Tier: 3
Examples: ["05555500", "00555550"]
Odds: 1 in 9
Price: $10-$500
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    if d:sub(1, 1) ~= "0" or d:sub(8, 8) ~= "0" then
        return {matched = false}
    end

    -- Find a run of 5+ identical non-zero digits; all other positions must be zero
    local runs = find_runs(d)
    for _, run in ipairs(runs) do
        if run.length >= 5 and run.digit ~= "0" then
            -- All other positions must be zero
            local all_others_zero = true
            for j = 1, 8 do
                local pos = j - 1  -- 0-indexed
                if pos < run.start or pos >= run.start + run.length then
                    if d:sub(j, j) ~= "0" then
                        all_others_zero = false
                        break
                    end
                end
            end
            if all_others_zero then
                return {
                    matched = true,
                    group_boxes = {
                        {from = run.start, to = run.start + run.length - 1, color = "gold", thickness = 3}
                    },
                    message = run.length .. " " .. run.digit .. "s stand-alone (CS-Stand Alone Quint)"
                }
            end
        end
    end

    return {matched = false}
end
