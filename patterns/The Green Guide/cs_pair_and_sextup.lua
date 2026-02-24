--[[
Pattern: CS_PAIR_AND_SEXTUP
DisplayName: CS-Pair and a Sextup
Description: A CS-Sextup (6 consecutive identical digits) plus a CS-Pair of a different digit, where the pair occupies positions 1–2 or 7–8. e.g., M 22666666 M or M 66666622 M.
BookRef: CS-450
Tier: 4
Examples: ["22666666", "66666622", "33555555", "55555533"]
Odds: 1 in 810
Price: $25-$500
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    local sext_digit, pair_digit = nil, nil
    for digit, cnt in pairs(counts) do
        if cnt == 6 then
            if sext_digit ~= nil then return {matched = false} end
            sext_digit = digit
        elseif cnt == 2 then
            if pair_digit ~= nil then return {matched = false} end
            pair_digit = digit
        else
            return {matched = false}
        end
    end

    if not sext_digit or not pair_digit then return {matched = false} end

    -- Sextup must be a consecutive 6-run
    local sext_run = nil
    for _, run in ipairs(find_runs(d)) do
        if run.digit == sext_digit and run.length >= 6 then
            sext_run = run
            break
        end
    end
    if not sext_run then return {matched = false} end

    -- Pair must be consecutive at positions 0–1 (start) or 6–7 (end) only
    local pair_positions = find_digit_positions(d, pair_digit)
    if pair_positions[2] - pair_positions[1] ~= 1 then return {matched = false} end
    local pair_start = pair_positions[1]
    if pair_start ~= 0 and pair_start ~= 6 then return {matched = false} end

    return {
        matched = true,
        group_boxes = {
            {from = sext_run.start, to = sext_run.start + sext_run.length - 1, color = "gold",   thickness = 3},
            {from = pair_start,     to = pair_start + 1,                        color = "orange", thickness = 3}
        },
        message = sext_digit .. "x6 sextup + " .. pair_digit .. pair_digit ..
                  " pair at " .. (pair_start == 0 and "start" or "end") .. " (CS-Pair and a Sextup)"
    }
end
