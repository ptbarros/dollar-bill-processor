--[[
Pattern: NICKS_FIVE_IN_A_ROW
DisplayName: 5 in a Row
Description: 5 consecutive identical digits
Tier: 4
Examples: ["00000123", "12300000", "11111234", "23411111", "99990123"]
--]]

function match(ctx)
    local s = ctx.digits

    for start = 1, 4 do
        local digit = s:sub(start, start)
        local all_same = true
        for i = start, start + 4 do
            if s:sub(i, i) ~= digit then
                all_same = false
                break
            end
        end

        if all_same then
            local extends_before = (start > 1 and s:sub(start - 1, start - 1) == digit)
            local extends_after = (start + 5 <= 8 and s:sub(start + 5, start + 5) == digit)

            if not extends_before and not extends_after then
                local positions = {}
                for i = start - 1, start + 3 do
                    table.insert(positions, i)
                end
                return {
                    matched = true,
                    message = "5 consecutive " .. digit .. "s",
                    highlights = {{positions = positions, color = "orange"}}
                }
            end
        end
    end

    return {matched = false}
end
