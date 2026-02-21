--[[
Pattern: CS_RANDOM_40AK
DisplayName: CS-Random 4OAK
Description: Four of the same digit anywhere in the serial as long as it does not make a CS-Quad (no run of 4+). e.g., M 41442435 M or M 14442435 M.
BookRef: CS-220
Tier: 6
Examples: ["41442435", "14442435", "44144144"]
Odds: 1 in 2,044
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    for digit, cnt in pairs(counts) do
        if cnt >= 4 then
            -- Verify no run of 4+ for this digit (must not be a CS-Quad)
            local runs = find_runs(d)
            local max_run = 0
            for _, run in ipairs(runs) do
                if run.digit == digit and run.length > max_run then
                    max_run = run.length
                end
            end

            if max_run < 4 then
                local positions = find_digit_positions(d, digit)
                return {
                    matched = true,
                    highlights = {{positions = positions, color = "orange"}},
                    message = cnt .. " scattered " .. digit .. "s (CS-Random 40AK)"
                }
            end
        end
    end

    return {matched = false}
end
