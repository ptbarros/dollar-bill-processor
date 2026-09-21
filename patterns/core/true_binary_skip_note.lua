--[[
Pattern: TRUE_BINARY_SKIP_NOTE
DisplayName: True Binary Alternator
Description: 0 and 1 strictly alternate across all 8 digits, 01010101 or 10101010, and nothing else.
BookRef: CS-1630
Tier: 1
Odds: 1 in 48,000,000 (2 per 96M)
Examples: ["01010101", "10101010"]
Price: $5-$200
--]]

function match(ctx)
    local d = ctx.digits

    if d ~= "01010101" and d ~= "10101010" then
        return {matched = false}
    end

    -- Alternating blue/cyan highlights
    local blue_pos = {}
    local cyan_pos = {}
    for i = 0, 7 do
        if i % 2 == 0 then
            table.insert(blue_pos, i)
        else
            table.insert(cyan_pos, i)
        end
    end

    return {
        matched = true,
        highlights = {
            {positions = blue_pos, color = "blue"},
            {positions = cyan_pos, color = "cyan"},
        },
        message = "CS-True Binary Skip Note: " .. d .. " (CS-1630)"
    }
end
