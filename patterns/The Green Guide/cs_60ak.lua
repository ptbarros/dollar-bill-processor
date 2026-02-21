--[[
Pattern: CS_60AK
DisplayName: CS-60AK (Scattered Sextup)
Description: Six of the same digit anywhere in the serial, with at least one separated from the others (no run of 6+). Grouped version is CS-Sextup (CS-400).
Tier: 3
Examples: ["11111211", "00000600", "11011111"]
Price: $50-$200
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    for digit, cnt in pairs(counts) do
        if cnt >= 6 then
            local max_run = 0
            local runs = find_runs(d)
            for _, run in ipairs(runs) do
                if run.digit == digit and run.length > max_run then
                    max_run = run.length
                end
            end

            if max_run < 6 then
                local positions = find_digit_positions(d, digit)
                return {
                    matched = true,
                    highlights = {{positions = positions, color = "orange"}},
                    message = cnt .. " scattered " .. digit .. "s (CS-60AK)"
                }
            end
        end
    end
    return {matched = false}
end
