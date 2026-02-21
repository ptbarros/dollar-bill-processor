--[[
Pattern: CS_QUINT_IN_PAIR
DisplayName: CS-Quint in a Pair
Description: A CS-Quint (5 consecutive) within a CS-20AK (pair of another digit surrounding it), plus one random digit. e.g., M 2555552x M.
BookRef: CS-380
Tier: 4
Examples: ["25555524", "52555520", "05555521"]
Odds: 1 in 720
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

    -- Quint digit must form a consecutive 5-run
    local quint_run = nil
    for _, run in ipairs(find_runs(d)) do
        if run.digit == quint_digit and run.length >= 5 then
            quint_run = run
            break
        end
    end
    if not quint_run then return {matched = false} end

    -- Pair digit must appear on BOTH sides of the quint run (surrounding it)
    local pair_positions = find_digit_positions(d, pair_digit)
    local has_before, has_after = false, false
    for _, pos in ipairs(pair_positions) do
        if pos < quint_run.start then has_before = true end
        if pos >= quint_run.start + quint_run.length then has_after = true end
    end
    if not has_before or not has_after then return {matched = false} end

    local single_positions = find_digit_positions(d, single_digit)

    return {
        matched = true,
        group_boxes = {
            {from = quint_run.start, to = quint_run.start + quint_run.length - 1, color = "gold", thickness = 3}
        },
        highlights = {
            {positions = pair_positions, color = "orange"},
            {positions = single_positions, color = "gray"}
        },
        connectors = {
            {from = pair_positions[1], to = pair_positions[2], color = "orange", style = "arc"}
        },
        message = quint_digit .. "x5 quint within " .. pair_digit .. " pair (CS-Quint in a Pair)"
    }
end
