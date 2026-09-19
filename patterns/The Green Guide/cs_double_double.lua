--[[
Pattern: CS_DOUBLE_DOUBLE
DisplayName: CS-Double Double
Description: Two digits, each appearing as two side-by-side pairs, alternating across the serial (e.g. 99·55·99·55).
BookRef: CS-280
Tier: 4
Examples: ["99559955", "11221122", "55885588"]
Odds: 1 in 630
Price: $25-$500
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    -- Must be exactly 2 distinct digits, each with count 4
    for _, cnt in pairs(counts) do
        if cnt ~= 4 then return {matched = false} end
    end
    -- Collect the two digits in reading order (first appearance) for deterministic colours.
    local digits_list = {}
    local seen = {}
    for i = 1, 8 do
        local ch = d:sub(i, i)
        if not seen[ch] then seen[ch] = true; table.insert(digits_list, ch) end
    end
    if #digits_list ~= 2 then return {matched = false} end

    -- Each digit must appear as exactly 2 separate runs, each of length 2 (no 4-run)
    local runs = find_runs(d)
    local digit_runs = {}
    for _, run in ipairs(runs) do
        if not digit_runs[run.digit] then digit_runs[run.digit] = {} end
        table.insert(digit_runs[run.digit], run)
    end

    for _, druns in pairs(digit_runs) do
        if #druns ~= 2 then return {matched = false} end
        for _, run in ipairs(druns) do
            if run.length ~= 2 then return {matched = false} end
        end
    end

    local d1, d2 = digits_list[1], digits_list[2]
    local d1_pos = find_digit_positions(d, d1)
    local d2_pos = find_digit_positions(d, d2)

    return {
        matched = true,
        highlights = {
            {positions = d1_pos, color = "orange"},
            {positions = d2_pos, color = "coral"}
        },
        message = "Split pairs: " .. d1 .. d1 .. ".." .. d1 .. d1 .. " + " .. d2 .. d2 .. ".." .. d2 .. d2 .. " (CS-Double Double)"
    }
end
