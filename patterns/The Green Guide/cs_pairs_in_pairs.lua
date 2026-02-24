--[[
Pattern: CS_PAIRS_IN_PAIRS
DisplayName: CS-Pairs in Pairs
Description: A CS-Pair (two adjacent identical digits) must be within a CS-2OAK (two non-adjacent identical digits of another value). The 2OAK digit must appear on both sides of the grouped pair. e.g., M 80085775 M (pair 55 within 2OAK of 8s: 8...8 surrounding 55). Uses all 8 positions with the two digit types.
BookRef: CS-80
Tier: 6
Examples: ["80085775", "85007758", "80055778"]
Odds: 1 in 2,520
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    -- Find the grouped pair digit and the surrounding 2OAK digit
    -- pair_digit: appears exactly 2 times and is adjacent
    -- oak_digit: appears exactly 2 times and is non-adjacent (surrounding the pair)
    local pair_digit, oak_digit = nil, nil
    local pair_pos1, pair_pos2, oak_pos1, oak_pos2

    for digit, cnt in pairs(counts) do
        if cnt == 2 then
            local positions = find_digit_positions(d, digit)
            if positions[2] - positions[1] == 1 then
                -- Grouped pair candidate
                if pair_digit == nil then
                    pair_digit = digit
                    pair_pos1 = positions[1]
                    pair_pos2 = positions[2]
                else
                    -- Multiple grouped pair candidates — check if one surrounds the other
                    -- Actually, allow: we need to find a grouped pair within a 2OAK
                    -- Reset — be flexible about which is which
                    -- We'll just collect all and test combinations
                end
            else
                -- Non-adjacent 2OAK candidate
                if oak_digit == nil then
                    oak_digit = digit
                    oak_pos1 = positions[1]
                    oak_pos2 = positions[2]
                end
            end
        elseif cnt ~= 1 then
            -- Digits with 3+ count break the pattern (unless 1-count fillers)
            -- For this pattern, we want exactly 2 types of 2-count digit + some singles
        end
    end

    -- Simple approach: look for any grouped pair that is surrounded by a 2OAK
    -- Collect all consecutive pairs
    local consecutive_pairs = {}
    for i = 1, #d - 1 do
        if d:sub(i, i) == d:sub(i+1, i+1) then
            local digit = d:sub(i, i)
            table.insert(consecutive_pairs, {digit = digit, start = i - 1, stop = i})
            -- skip the pair
        end
    end

    -- For each grouped pair, check if there's a 2OAK surrounding it
    for _, cp in ipairs(consecutive_pairs) do
        -- Find positions of any digit that appears on both sides of this pair
        for digit, cnt in pairs(counts) do
            if digit ~= cp.digit and cnt == 2 then
                local positions = find_digit_positions(d, digit)
                local has_before = positions[1] < cp.start
                local has_after = positions[2] > cp.stop
                if has_before and has_after then
                    -- The 2OAK surrounds the pair
                    local oak_positions = positions
                    return {
                        matched = true,
                        group_boxes = {
                            {from = cp.start, to = cp.stop, color = "orange", thickness = 3}
                        },
                        highlights = {
                            {positions = oak_positions, color = "gold"}
                        },
                        connectors = {
                            {from = oak_positions[1], to = oak_positions[2], color = "gold", style = "arc"}
                        },
                        message = cp.digit .. cp.digit .. " pair within " .. digit .. " 2OAK (CS-Pairs in Pairs)"
                    }
                end
            end
        end
    end

    return {matched = false}
end
