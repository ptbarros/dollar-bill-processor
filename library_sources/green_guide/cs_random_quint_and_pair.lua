--[[
Pattern: CS_RANDOM_QUINT_AND_PAIR
DisplayName: CS-Random Quint and Pair
Description: One digit appears five times scattered about, plus a side-by-side pair of another digit and one leftover (e.g. 5552·5251).
BookRef: CS-400
Tier: 6
Examples: ["55525251", "55050515", "51553551"]
Odds: 1 in 120,960
Price: $0.25
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    -- Must have exactly 3 distinct digits with counts {5, 2, 1}
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

    -- Quint digit must NOT have a 5-run (must be scattered = "random")
    for _, run in ipairs(find_runs(d)) do
        if run.digit == quint_digit and run.length >= 5 then
            return {matched = false}  -- That's CS-Quint in a Pair (CS-380)
        end
    end

    local quint_positions = find_digit_positions(d, quint_digit)
    local pair_positions = find_digit_positions(d, pair_digit)
    local single_positions = find_digit_positions(d, single_digit)

    return {
        matched = true,
        highlights = {
            {positions = quint_positions, color = "gold"},
            {positions = pair_positions, color = "orange"},
            {positions = single_positions, color = "charcoal", style = "x"}
        },
        message = quint_digit .. "x5 scattered + " .. pair_digit .. " pair (CS-Random Quint and Pair)"
    }
end
