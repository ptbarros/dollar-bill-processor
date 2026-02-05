--[[
Pattern: FIVE_OF_A_KIND
DisplayName: 5 of a Kind
Description: Exactly 5 of the same digit
Tier: 4
Examples: ["00000123", "11111234", "22222012", "55555123", "99999012"]
--]]

function match(ctx)
    local s = ctx.digits
    local counts = {}

    for i = 1, 8 do
        local d = s:sub(i, i)
        counts[d] = (counts[d] or 0) + 1
    end

    for digit, count in pairs(counts) do
        if count == 5 then
            local positions = {}
            for i = 1, 8 do
                if s:sub(i, i) == digit then
                    table.insert(positions, i - 1)
                end
            end
            return {
                matched = true,
                message = "5 of a kind: " .. digit,
                highlights = {{positions = positions, color = "gold"}}
            }
        end
    end

    return {matched = false}
end
