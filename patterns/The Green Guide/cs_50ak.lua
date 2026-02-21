--[[
Pattern: CS_50AK
DisplayName: CS-5OAK
Description: Five of the same digit anywhere in the serial, with at least one separated from the others (no run of 5+). Grouped version is CS-Quint (CS-300).
BookRef: CS-360
Tier: 4
Examples: ["11112111", "00005000", "10101010"]
Price: $20-$60
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    for digit, cnt in pairs(counts) do
        if cnt >= 5 then
            local max_run = 0
            local runs = find_runs(d)
            for _, run in ipairs(runs) do
                if run.digit == digit and run.length > max_run then
                    max_run = run.length
                end
            end

            if max_run < 5 then
                local positions = find_digit_positions(d, digit)
                return {
                    matched = true,
                    highlights = {{positions = positions, color = "orange"}},
                    message = cnt .. " scattered " .. digit .. "s (CS-50AK)"
                }
            end
        end
    end
    return {matched = false}
end
