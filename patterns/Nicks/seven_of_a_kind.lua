--[[
Pattern: SEVEN_OF_A_KIND
DisplayName: 7 of a Kind
Description: Exactly 7 of the same digit
Tier: 2
Examples: ["00000001", "11111112", "22222220", "55555559", "99999990"]
--]]

function match(ctx)
    local s = ctx.digits
    local counts = {}

    for i = 1, 8 do
        local d = s:sub(i, i)
        counts[d] = (counts[d] or 0) + 1
    end

    for digit, count in pairs(counts) do
        if count == 7 then
            local positions = {}
            for i = 1, 8 do
                if s:sub(i, i) == digit then
                    table.insert(positions, i - 1)
                end
            end
            return {
                matched = true,
                message = "7 of a kind: " .. digit,
                highlights = {{positions = positions, color = "gold"}}
            }
        end
    end

    return {matched = false}
end
