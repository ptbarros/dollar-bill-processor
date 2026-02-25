--[[
Pattern: CS_40AK
DisplayName: CS-4OAK
Description: Any four or more of the same digit anywhere in the serial, in any arrangement (grouped or scattered). More specific patterns: CS-Quad (CS-200) for consecutive, CS-Random 4OAK (CS-210) for scattered.
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
