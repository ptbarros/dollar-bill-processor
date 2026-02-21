--[[
Pattern: CS_RANDOM_PAIR_IN_SEXTUP
DisplayName: CS-Random Pair in a Sextup
Description: A CS-20AK (2 scattered, non-adjacent digits) with a CS-60AK (6 of another digit). Exactly two digits using all 8 positions. Also a CS-Binary. e.g., M 26666266 M.
BookRef: CS-470
Tier: 3
Examples: ["26666266", "62666626", "66266626"]
Odds: 1 in 90
Price: $25-$500
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    -- Must be exactly 2 distinct digits with counts {2, 6}
    local pair_digit, sextup_digit = nil, nil
    for digit, cnt in pairs(counts) do
        if cnt == 2 then
            if pair_digit ~= nil then return {matched = false} end
            pair_digit = digit
        elseif cnt == 6 then
            if sextup_digit ~= nil then return {matched = false} end
            sextup_digit = digit
        else
            return {matched = false}
        end
    end

    if not pair_digit or not sextup_digit then return {matched = false} end

    -- Pair digit must NOT be adjacent (no 2-run) — that would be CS-Pair in Sextup (CS-460)
    local pair_positions = find_digit_positions(d, pair_digit)
    if pair_positions[2] - pair_positions[1] == 1 then
        return {matched = false}  -- consecutive = CS-460
    end

    local sextup_positions = find_digit_positions(d, sextup_digit)

    return {
        matched = true,
        highlights = {
            {positions = sextup_positions, color = "gold"},
            {positions = pair_positions, color = "orange"}
        },
        connectors = {
            {from = pair_positions[1], to = pair_positions[2], color = "orange", style = "arc"}
        },
        message = pair_digit .. " random pair within " .. sextup_digit .. "x6 (CS-Random Pair in Sextup)"
    }
end
