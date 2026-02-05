--[[
Pattern: NICKS_SIX_IN_A_ROW
DisplayName: 6 in a Row
Description: 6 consecutive identical digits
Tier: 3
Examples: ["00000012", "12000000", "11111123", "23111111", "99999012"]
--]]

function match(ctx)
    local s = ctx.digits

    for start = 1, 3 do
        local digit = s:sub(start, start)
        local all_same = true
        for i = start, start + 5 do
            if s:sub(i, i) ~= digit then
                all_same = false
                break
            end
        end

        if all_same then
            -- Check it doesn't extend to 7+
            local extends_before = (start > 1 and s:sub(start - 1, start - 1) == digit)
            local extends_after = (start + 6 <= 8 and s:sub(start + 6, start + 6) == digit)

            if not extends_before and not extends_after then
                local positions = {}
                for i = start - 1, start + 4 do
                    table.insert(positions, i)
                end
                return {
                    matched = true,
                    message = "6 consecutive " .. digit .. "s",
                    highlights = {{positions = positions, color = "orange"}}
                }
            end
        end
    end

    return {matched = false}
end
