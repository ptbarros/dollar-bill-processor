--[[
Pattern: CS_30AK
DisplayName: CS-3OAK
Description: The same digit shows up three times, scattered — at least one is split off from the other two (e.g. 33·xxx·3·xx).
BookRef: CS-110
Tier: 6
Examples: ["30130055", "11011234", "10102034"]
Price: $5-$15
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    for digit, cnt in pairs(counts) do
        if cnt >= 3 then
            -- Find the max consecutive run length for this digit
            local max_run = 0
            local runs = find_runs(d)
            for _, run in ipairs(runs) do
                if run.digit == digit and run.length > max_run then
                    max_run = run.length
                end
            end

            -- Scattered: count >= 3 but no run of 3+ for this digit
            if max_run < 3 then
                local positions = find_digit_positions(d, digit)
                return {
                    matched = true,
                    highlights = {{positions = positions, color = "orange"}},
                    message = cnt .. " scattered " .. digit .. "s (CS-3OAK)"
                }
            end
        end
    end
    return {matched = false}
end
