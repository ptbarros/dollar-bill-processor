--[[
Pattern: CS_PAIR_IN_SEXTUP
DisplayName: CS-Pair in a Sextup
Description: A CS-Pair (2 consecutive same digit) within a CS-60AK (6 of another digit surrounding it). Exactly two digits using all 8 positions. Also a CS-Binary. e.g., M 62266666 M or M 66226666 M.
BookRef: CS-460
Tier: 3
Examples: ["62266666", "66226666", "66622666"]
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

    -- Pair digit must form a consecutive 2-run (grouped pair)
    local pair_run = nil
    for _, run in ipairs(find_runs(d)) do
        if run.digit == pair_digit and run.length >= 2 then
            pair_run = run
            break
        end
    end
    if not pair_run then return {matched = false} end  -- scattered, not grouped

    -- Sextup digit must appear on BOTH sides of the pair run (surrounding it)
    local sextup_positions = find_digit_positions(d, sextup_digit)
    local has_before, has_after = false, false
    for _, pos in ipairs(sextup_positions) do
        if pos < pair_run.start then has_before = true end
        if pos >= pair_run.start + pair_run.length then has_after = true end
    end
    if not has_before or not has_after then return {matched = false} end

    return {
        matched = true,
        highlights = {
            {positions = sextup_positions, color = "gold"},
            {positions = {pair_run.start, pair_run.start + 1}, color = "orange"}
        },
        group_boxes = {
            {from = pair_run.start, to = pair_run.start + 1, color = "orange", thickness = 3}
        },
        message = pair_digit .. pair_digit .. " pair within " .. sextup_digit .. "x6 (CS-Pair in Sextup)"
    }
end
