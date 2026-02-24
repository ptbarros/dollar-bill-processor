--[[
Pattern: CS_RANDOM_PAIRS_IN_PAIRS
DisplayName: CS-Random Pairs in Pairs
Description: Two CS-2OAKs where one is inside the other — one 2OAK digit appears on both sides of the other 2OAK digit. Neither pair needs to be grouped (both can be scattered). e.g., M xx2x2xxx M with 5s on either side of the 2s.
BookRef: CS-90
Tier: 7
Examples: ["52302352", "29002293", "83133831"]
Odds: 1 in 1,680
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    -- Collect all digits with exactly 2 occurrences
    local two_count_digits = {}
    for digit, cnt in pairs(counts) do
        if cnt == 2 then
            table.insert(two_count_digits, digit)
        end
    end

    -- Need at least 2 digits with exactly 2 occurrences
    if #two_count_digits < 2 then
        return {matched = false}
    end

    -- Check each pair of 2OAK digits: does one surround the other?
    table.sort(two_count_digits)
    for i = 1, #two_count_digits do
        for j = i + 1, #two_count_digits do
            local da = two_count_digits[i]
            local db = two_count_digits[j]
            local pos_a = find_digit_positions(d, da)
            local pos_b = find_digit_positions(d, db)

            -- Check if da surrounds db: pos_a[1] < pos_b[1] and pos_a[2] > pos_b[2]
            -- Also check if pos_a[1] < pos_b[2] and pos_a[2] > pos_b[1] (interleaved interior)
            -- Book says "one 2OAK inside another" — outer must have both positions outside inner
            local a_surrounds_b = pos_a[1] < pos_b[1] and pos_a[2] > pos_b[2]
            local b_surrounds_a = pos_b[1] < pos_a[1] and pos_b[2] > pos_a[2]

            if a_surrounds_b then
                -- da is outer, db is inner
                local color_outer = "gold"
                local color_inner = "orange"
                return {
                    matched = true,
                    highlights = {
                        {positions = pos_a, color = color_outer},
                        {positions = pos_b, color = color_inner}
                    },
                    connectors = {
                        {from = pos_a[1], to = pos_a[2], color = color_outer, style = "arc"},
                        {from = pos_b[1], to = pos_b[2], color = color_inner, style = "arc"}
                    },
                    message = db .. " 2OAK inside " .. da .. " 2OAK (CS-Random Pairs in Pairs)"
                }
            elseif b_surrounds_a then
                -- db is outer, da is inner
                local color_outer = "gold"
                local color_inner = "orange"
                return {
                    matched = true,
                    highlights = {
                        {positions = pos_b, color = color_outer},
                        {positions = pos_a, color = color_inner}
                    },
                    connectors = {
                        {from = pos_b[1], to = pos_b[2], color = color_outer, style = "arc"},
                        {from = pos_a[1], to = pos_a[2], color = color_inner, style = "arc"}
                    },
                    message = da .. " 2OAK inside " .. db .. " 2OAK (CS-Random Pairs in Pairs)"
                }
            end
        end
    end

    return {matched = false}
end
