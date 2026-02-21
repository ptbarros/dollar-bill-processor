--[[
Pattern: CS_TRIPLE_IN_QUINT
DisplayName: CS-Triple in a Quint
Description: A CS-Triple (3 consecutive) within a CS-50AK (5 of another digit) where the 5-digit must bookend the serial (appear at positions 1 and 8). Exactly two digits using all 8 positions. Also a CS-Binary. e.g., M 53335555 M or M 55333555 M.
BookRef: CS-420
Tier: 3
Examples: ["53335555", "55333555", "55533355"]
Odds: 1 in 90
Price: $25-$500
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    -- Must be exactly 2 distinct digits with counts {3, 5}
    local triple_digit, quint_digit = nil, nil
    for digit, cnt in pairs(counts) do
        if cnt == 3 then
            if triple_digit ~= nil then return {matched = false} end
            triple_digit = digit
        elseif cnt == 5 then
            if quint_digit ~= nil then return {matched = false} end
            quint_digit = digit
        else
            return {matched = false}
        end
    end

    if not triple_digit or not quint_digit then return {matched = false} end

    -- Triple digit must form a consecutive 3-run
    local triple_run = nil
    for _, run in ipairs(find_runs(d)) do
        if run.digit == triple_digit and run.length >= 3 then
            triple_run = run
            break
        end
    end
    if not triple_run then return {matched = false} end

    -- Quint digit (50AK) must appear at BOTH position 1 AND position 8 (bookending)
    if d:sub(1, 1) ~= quint_digit or d:sub(8, 8) ~= quint_digit then
        return {matched = false}
    end

    local quint_positions = find_digit_positions(d, quint_digit)

    return {
        matched = true,
        group_boxes = {
            {from = triple_run.start, to = triple_run.start + triple_run.length - 1, color = "orange", thickness = 3}
        },
        highlights = {
            {positions = quint_positions, color = "gold"}
        },
        connectors = {
            {from = 0, to = 7, color = "gold", style = "arc"}
        },
        message = triple_digit .. "x3 triple within " .. quint_digit .. "x5 bookending quint (CS-Triple in Quint)"
    }
end
