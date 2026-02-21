--[[
Pattern: CS_LUCKY_SEVEN_RADAR
DisplayName: CS-Lucky Seven Radar
Description: A CS-Quint or CS-50AK (5+ of same digit) within a CS-20AK (same different digit at both ends): e.g., 25555552. The outer digit bookends a run of 6 (or 5) identical inner digits.
BookRef: CS-1350
Tier: 2
Examples: ["25555552", "13333331", "70000007"]
Price: $0.50-$5
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Pattern 1: Xd₁d₁d₁d₁d₁d₁X (outer pair wraps inner sextup) = A+BBBBBB+A
    -- This is actually CS-Super Radar (ABBBBBBA). Lucky Seven Radar per plan = CS-50AK within CS-20AK

    -- Check XYYYYYYYX: first == last (pair), middle 6 are all same and ≠ outer
    local outer = d:sub(1, 1)
    if d:sub(8, 8) == outer then
        -- Check middle 6 (positions 2-7 in Lua)
        local inner = d:sub(2, 2)
        if inner ~= outer then
            local all_inner = true
            for i = 3, 7 do
                if d:sub(i, i) ~= inner then
                    all_inner = false
                    break
                end
            end
            if all_inner then
                return {
                    matched = true,
                    highlights = {
                        {positions = {0, 7}, color = "orange"},
                        {positions = {1, 2, 3, 4, 5, 6}, color = "gold"},
                    },
                    group_boxes = {{from = 1, to = 6, color = "gold", thickness = 3}},
                    connectors = {{from = 0, to = 7, color = "orange", style = "arc"}},
                    message = outer .. " wraps sextup of " .. inner .. "s (CS-Lucky Seven Radar CS-1350)"
                }
            end
        end
    end

    -- Pattern 2: XYYYYYYX (outer pair, inner quint + 1 different = CS-50AK style)
    -- e.g., 25555552 where first 2 positions = outer, inner 4 = same, but let's check
    -- Actually plan says "A+BBBBBB+A" is the primary example. Let's also check XXBBBBBXX (pair bookending quint)
    -- e.g., 22555552: outer pair 22, inner 5 of 5s, trailing 2
    -- Quint within 20AK: need positions 0 and 7 same (20AK), AND a quint of a different digit somewhere inside

    if d:sub(1, 1) == d:sub(8, 8) then
        local inner_str = d:sub(2, 7)
        local quint = has_n_consecutive(inner_str, 5)
        if quint then
            local outer2 = d:sub(1, 1)
            if quint.digit ~= outer2 then
                local inner_positions = {}
                for i = 1, 6 do
                    if inner_str:sub(i, i) == quint.digit then
                        table.insert(inner_positions, i)  -- 0-indexed: inner offset + 1
                    end
                end
                -- Adjust to full-string 0-indexed positions
                local adj_positions = {}
                for _, p in ipairs(inner_positions) do
                    table.insert(adj_positions, p)  -- positions 1-6 in 0-indexed
                end
                return {
                    matched = true,
                    highlights = {
                        {positions = {0, 7}, color = "orange"},
                        {positions = adj_positions, color = "gold"},
                    },
                    message = outer2 .. " bookends quint of " .. quint.digit .. "s (CS-Lucky Seven Radar CS-1350)"
                }
            end
        end
    end

    return {matched = false}
end
