--[[
Pattern: 1959
DisplayName: 1959
Description: Year note 1959
Tier: 5
Odds: 1 in 1,967 (48,799 per 96M)
Examples: ["19590000", "00195900", "00001959"]
--]]

function match(ctx)
    local pos = ctx.digits:find("1959", 1, true)
    if pos then
        local positions = {}
        for i = 0, 3 do
            table.insert(positions, pos - 1 + i)
        end
        return {
            matched = true,
            highlights = {{positions = positions, color = "orange"}},
            message = "Year note 1959"
        }
    end
    return {matched = false}
end