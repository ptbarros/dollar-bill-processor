--[[
Pattern: CS_QUINT_IN_TRIPLE
DisplayName: CS-Quint in a Triple
Description: A CS-Quint (5 consecutive) within a CS-30AK (3 of another digit surrounding it). Exactly two digits using all 8 positions. Also a CS-Binary. e.g., M 35555533 M or M 33555553 M.
BookRef: CS-410
Tier: 3
Examples: ["35555533", "33555553", "53555535"]
Odds: 1 in 90
Price: $25-$500
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    -- Must be exactly 2 distinct digits with counts {5, 3}
    local quint_digit, triple_digit = nil, nil
    for digit, cnt in pairs(counts) do
        if cnt == 5 then
            if quint_digit ~= nil then return {matched = false} end
            quint_digit = digit
        elseif cnt == 3 then
            if triple_digit ~= nil then return {matched = false} end
            triple_digit = digit
        else
            return {matched = false}
        end
    end

    if not quint_digit or not triple_digit then return {matched = false} end

    -- Quint digit must form a consecutive 5-run
    local quint_run = nil
    for _, run in ipairs(find_runs(d)) do
        if run.digit == quint_digit and run.length >= 5 then
            quint_run = run
            break
        end
    end
    if not quint_run then return {matched = false} end

    -- Triple digit must appear on BOTH sides of the quint run
    local triple_positions = find_digit_positions(d, triple_digit)
    local has_before, has_after = false, false
    for _, pos in ipairs(triple_positions) do
        if pos < quint_run.start then has_before = true end
        if pos >= quint_run.start + quint_run.length then has_after = true end
    end
    if not has_before or not has_after then return {matched = false} end

    return {
        matched = true,
        group_boxes = {
            {from = quint_run.start, to = quint_run.start + quint_run.length - 1, color = "gold", thickness = 3}
        },
        highlights = {
            {positions = triple_positions, color = "orange"}
        },
        message = quint_digit .. "x5 quint within " .. triple_digit .. "x3 triple (CS-Quint in Triple)"
    }
end
