--[[
Pattern: CS_DOUBLE_40AK
DisplayName: CS-Random Double 4OAK
Description: Exactly two distinct digits, each appearing 4 times, both scattered (no 4-run for either digit). Contrast: CS-250 has one digit in a 4-run; CS-280 has two digits in paired 2-runs.
BookRef: CS-240
Tier: 4
Examples: ["12121212", "11211221", "10100101"]
Odds: 1 in 35
Price: $10-$50
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    -- Exactly 2 distinct digits, each appearing exactly 4 times
    local digits = {}
    for digit, cnt in pairs(counts) do
        if cnt ~= 4 then return {matched = false} end
        table.insert(digits, digit)
    end
    if #digits ~= 2 then return {matched = false} end

    -- Neither digit may have a run of 4+ (both must be scattered)
    local runs = find_runs(d)
    for _, run in ipairs(runs) do
        if run.length >= 4 then return {matched = false} end
    end

    local d1 = digits[1]
    local d2 = digits[2]
    local pos1 = find_digit_positions(d, d1)
    local pos2 = find_digit_positions(d, d2)

    return {
        matched = true,
        highlights = {
            {positions = pos1, color = "orange"},
            {positions = pos2, color = "cyan"}
        },
        message = "4×" .. d1 .. " + 4×" .. d2 .. ", both scattered (CS-Double 40AK)"
    }
end
