--[[
Pattern: CS_40AK
DisplayName: CS-4OAK
Description: The same digit appears four or more times, anywhere in the serial, grouped or scattered (e.g. 44·4·x·4·xx).
BookRef: CS-190
Tier: 5
Examples: ["10101012", "00303030", "01010101", "44441234", "11112345"]
Price: $10-$30
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    for digit, cnt in pairs(counts) do
        if cnt >= 4 then
            local positions = find_digit_positions(d, digit)
            return {
                matched = true,
                highlights = {{positions = positions, color = "orange"}},
                message = cnt .. "× " .. digit .. " (CS-4OAK)"
            }
        end
    end
    return {matched = false}
end
