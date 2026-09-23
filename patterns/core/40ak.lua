--[[
Pattern: 40AK
DisplayName: 4 of a Kind
Description: The same digit appears exactly four times, anywhere in the serial, grouped or scattered (e.g. 44·4·x·4·xx).
Tier: 9
Odds: 1 in 22 (4,363,755 per 96M)
Examples: ["10101012", "01010101", "11112345", "70717273", "17171712"]
Price: $10-$30
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    -- Exactly four of a kind (Ed review): five-plus is owned by 5/6/7 of a Kind.
    for digit, cnt in pairs(counts) do
        if cnt == 4 then
            local positions = find_digit_positions(d, digit)
            return {
                matched = true,
                highlights = {{positions = positions, color = "orange"}},
                message = "4× " .. digit .. " (4 of a Kind)"
            }
        end
    end
    return {matched = false}
end
