--[[
Pattern: CS_70AK
DisplayName: CS-7OAK
Description: The same digit fills seven of the eight spots, with the odd one out breaking up the run so they're not all in a row (e.g. 1111·2·111).
BookRef: CS-490
Tier: 2
Examples: ["11111121", "00000100", "22222202"]
Price: $200-$800
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    for digit, cnt in pairs(counts) do
        if cnt >= 7 then
            local max_run = 0
            local runs = find_runs(d)
            for _, run in ipairs(runs) do
                if run.digit == digit and run.length > max_run then
                    max_run = run.length
                end
            end

            -- Scattered: count >= 7 but no run of 7+ for this digit
            if max_run < 7 then
                local positions = find_digit_positions(d, digit)
                return {
                    matched = true,
                    highlights = {{positions = positions, color = "orange"}},
                    message = cnt .. " scattered " .. digit .. "s (CS-70AK)"
                }
            end
        end
    end
    return {matched = false}
end
