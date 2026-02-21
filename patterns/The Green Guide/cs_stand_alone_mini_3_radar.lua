--[[
Pattern: CS_STAND_ALONE_MINI_3_RADAR
DisplayName: CS-Stand Alone Mini 3 Radar
Description: An AXA palindrome (A ≠ X, A ≠ 0, X ≠ 0) surrounded by zeros on both sides. e.g., M 00121000 M or M 01210000 M.
BookRef: CS-1730
Tier: 4
Examples: ["00121000", "01210000", "00012100"]
Odds: 1 in 1,500
Price: $10-$75
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Must start and end with 0 (zeros on both sides)
    if d:sub(1, 1) ~= "0" or d:sub(8, 8) ~= "0" then
        return {matched = false}
    end

    -- Scan for AXA at positions 2–4, 3–5, 4–6, or 5–7 (1-indexed)
    for start = 2, 5 do
        local a = d:sub(start,     start)
        local x = d:sub(start + 1, start + 1)
        local a2 = d:sub(start + 2, start + 2)

        if a == a2 and a ~= x and a ~= "0" and x ~= "0" then
            -- All other positions must be 0
            local all_others_zero = true
            for j = 1, 8 do
                if j < start or j > start + 2 then
                    if d:sub(j, j) ~= "0" then
                        all_others_zero = false
                        break
                    end
                end
            end
            if all_others_zero then
                local base = start - 1  -- 0-indexed
                return {
                    matched = true,
                    group_boxes = {
                        {from = base, to = base + 2, color = "orange", thickness = 3}
                    },
                    connectors = {
                        {from = base, to = base + 2, color = "orange", style = "arc"}
                    },
                    highlights = {
                        {positions = {base, base + 2}, color = "orange"},
                        {positions = {base + 1},       color = "coral"}
                    },
                    message = a .. x .. a .. " stand-alone mini-3 radar at position " .. start .. " (CS-Stand Alone Mini 3 Radar)"
                }
            end
        end
    end

    return {matched = false}
end
