--[[
Pattern: CS_50AK
DisplayName: CS-5OAK
Description: The same digit appears five times, scattered so they don't all sit in one solid block (e.g. 11·x·1·x·1·1).
BookRef: CS-360
Tier: 4
Examples: ["11112111", "00005000", "11011011"]
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
