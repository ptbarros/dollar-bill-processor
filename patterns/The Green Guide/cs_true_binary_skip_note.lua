--[[
Pattern: CS_TRUE_BINARY_SKIP_NOTE
DisplayName: CS-True Binary Skip Note
Description: Just 0s and 1s taking strict turns the whole way across — either 01010101 or 10101010.
BookRef: CS-1630
Tier: 2
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
