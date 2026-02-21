--[[
Pattern: CS_HIGH_NOTES
DisplayName: CS-High Notes
Description: Serial number >= 99990000 (four or more leading 9s). The more leading 9s, the rarer. e.g., M 9999xxxx M.
BookRef: CS-1970
Tier: 4
Examples: ["99990000", "99999999", "99991234"]
Odds: 1 in 9,000
Price: $0-$1,000+
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Must start with at least four 9s
    if d:sub(1, 4) ~= "9999" then
        return {matched = false}
    end

    -- Count leading 9s for message
    local leading_nines = 0
    for i = 1, 8 do
        if d:sub(i, i) == "9" then
            leading_nines = leading_nines + 1
        else
            break
        end
    end

    -- Highlight the leading 9s
    local nine_positions = {}
    for i = 0, leading_nines - 1 do
        table.insert(nine_positions, i)
    end

    return {
        matched = true,
        highlights = {
            {positions = nine_positions, color = "gold"}
        },
        message = leading_nines .. " leading 9s — CS-High Note (CS-1970)"
    }
end
