--[[
Pattern: CS_40AK
DisplayName: CS-4OAK
Description: Four of the same digit anywhere in the serial, with at least one separated from the others (no run of 4+). Grouped version is CS-Quad (CS-200).
BookRef: CS-190
Tier: 5
Examples: ["10101012", "00303030", "01010101"]
Price: $10-$30
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    for digit, cnt in pairs(counts) do
        if cnt >= 4 then
            local max_run = 0
            local runs = find_runs(d)
            for _, run in ipairs(runs) do
                if run.digit == digit and run.length > max_run then
                    max_run = run.length
                end
            end

            -- Scattered: count >= 4 but no run of 4+ for this digit
            if max_run < 4 then
                local positions = find_digit_positions(d, digit)
                return {
                    matched = true,
                    highlights = {{positions = positions, color = "orange"}},
                    message = cnt .. " scattered " .. digit .. "s (CS-4OAK)"
                }
            end
        end
    end
    return {matched = false}
end
