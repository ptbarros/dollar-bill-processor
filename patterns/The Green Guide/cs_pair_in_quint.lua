--[[
Pattern: CS_PAIR_IN_QUINT
DisplayName: CS-Pair in a Quint
Description: Five occurrences of one digit surround a consecutive pair of another digit, with one remaining digit anywhere. The quint digit must appear on both sides of the pair. e.g., M x5225555 M.
BookRef: CS-390
Tier: 5
Examples: ["05225555", "05522555", "52255550", "55225515", "52255155"]
Odds: 1 in 720
Price: $0.25
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    local quint_digit, pair_digit, single_digit = nil, nil, nil
    for digit, cnt in pairs(counts) do
        if cnt == 5 then
            if quint_digit ~= nil then return {matched = false} end
            quint_digit = digit
        elseif cnt == 2 then
            if pair_digit ~= nil then return {matched = false} end
            pair_digit = digit
        elseif cnt == 1 then
            if single_digit ~= nil then return {matched = false} end
            single_digit = digit
        else
            return {matched = false}
        end
    end

    if not quint_digit or not pair_digit or not single_digit then
        return {matched = false}
    end

    -- Pair must be consecutive (grouped)
    local pair_positions = find_digit_positions(d, pair_digit)
    if pair_positions[2] - pair_positions[1] ~= 1 then return {matched = false} end

    local single_positions = find_digit_positions(d, single_digit)

    -- Quint digit must appear on both sides of the pair
    local quint_positions = find_digit_positions(d, quint_digit)
    local pair_min = pair_positions[1]
    local pair_max = pair_positions[2]
    local has_before, has_after = false, false
    for _, pos in ipairs(quint_positions) do
        if pos < pair_min then has_before = true end
        if pos > pair_max then has_after = true end
    end
    if not has_before or not has_after then return {matched = false} end

    return {
        matched = true,
        highlights = {
            {positions = quint_positions,  color = "gold"},
            {positions = pair_positions,   color = "orange"},
            {positions = single_positions, color = "gray"}
        },
        connectors = {
            {from = pair_positions[1], to = pair_positions[2], color = "orange", style = "bracket"}
        },
        message = quint_digit .. "x5 surrounds " .. pair_digit .. pair_digit .. " pair (CS-Pair in a Quint)"
    }
end
