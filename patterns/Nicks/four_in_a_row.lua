--[[
Pattern: FOUR_IN_A_ROW
DisplayName: 4 in a Row
Description: 4 consecutive identical digits (quad)
Tier: 5
Examples: ["00001234", "12340000", "11112345", "23451111", "99901234"]
--]]

function match(ctx)
    local s = ctx.digits

    for start = 1, 5 do
        local digit = s:sub(start, start)
        local all_same = true
        for i = start, start + 3 do
            if s:sub(i, i) ~= digit then
                all_same = false
                break
            end
        end

        if all_same then
            local extends_before = (start > 1 and s:sub(start - 1, start - 1) == digit)
            local extends_after = (start + 4 <= 8 and s:sub(start + 4, start + 4) == digit)

            if not extends_before and not extends_after then
                local positions = {}
                for i = start - 1, start + 2 do
                    table.insert(positions, i)
                end
                return {
                    matched = true,
                    message = "Quad " .. digit .. "s",
                    highlights = {{positions = positions, color = "orange"}}
                }
            end
        end
    end

    return {matched = false}
end
