--[[
Pattern: CS_60AK
DisplayName: CS-6OAK
Description: The same digit appears six times, scattered so they don't all sit in one solid block (e.g. 1·1·1·x·1·1·1).
BookRef: CS-430
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
