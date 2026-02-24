--[[
Pattern: CS_RANDOM_TRIPLE_IN_QUAD
DisplayName: CS-Random Triple in Quad
Description: A 3OAK (three scattered instances of one digit) and a 4OAK (four scattered instances of another digit), where the 3OAK digit is surrounded by the 4OAK digit (the 4OAK has instances on both sides of all three 3OAK positions). Neither needs to be consecutive. e.g., CS-30AK and CS-40AK in any position with the 30AK surrounded.
BookRef: CS-300
Tier: 5
Examples: ["14141413", "21212123", "31313134"]
Odds: 1 in 4,320
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    local triple_digit, quad_digit = nil, nil

    for digit, cnt in pairs(counts) do
        if cnt == 3 then
            if triple_digit ~= nil then return {matched = false} end
            triple_digit = digit
        elseif cnt == 4 then
            if quad_digit ~= nil then return {matched = false} end
            quad_digit = digit
        elseif cnt == 1 then
            -- extra single digit: 3+4+1=8, valid if only one single
        else
            return {matched = false}
        end
    end

    if not triple_digit or not quad_digit then
        return {matched = false}
    end

    -- Verify 3+4=7 or 3+4+1=8 (allow one extra single)
    local total_accounted = 3 + 4
    local remaining = 8 - total_accounted
    -- Check remaining digits are all the same and count is 1 (one single)
    if remaining ~= 0 and remaining ~= 1 then
        return {matched = false}
    end

    -- Triple must be scattered (no consecutive run of 3) — otherwise it's CS-290 (Triple in Quad)
    for _, run in ipairs(find_runs(d)) do
        if run.digit == triple_digit and run.length >= 3 then
            return {matched = false}
        end
    end

    -- The 4OAK must NOT form a consecutive run of 4
    for _, run in ipairs(find_runs(d)) do
        if run.digit == quad_digit and run.length >= 4 then
            return {matched = false}
        end
    end

    -- The 4OAK digit must appear on BOTH sides of the full span of 3OAK positions
    local triple_pos = find_digit_positions(d, triple_digit)
    local quad_pos = find_digit_positions(d, quad_digit)

    local triple_min = triple_pos[1]
    local triple_max = triple_pos[#triple_pos]

    local has_before = false
    local has_after = false
    for _, pos in ipairs(quad_pos) do
        if pos < triple_min then has_before = true end
        if pos > triple_max then has_after = true end
    end

    if not has_before or not has_after then
        return {matched = false}
    end

    return {
        matched = true,
        highlights = {
            {positions = triple_pos, color = "orange"},
            {positions = quad_pos, color = "gold"}
        },
        message = triple_digit .. "x3 within " .. quad_digit .. "x4 scattered (CS-Random Triple in Quad)"
    }
end
