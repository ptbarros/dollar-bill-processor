--[[
Pattern: NICKS_SEVEN_IN_A_ROW
DisplayName: 7 in a Row
Description: 7 consecutive identical digits
Tier: 2
Examples: ["00000001", "10000000", "11111112", "21111111", "99999990"]
--]]

function match(ctx)
    local s = ctx.digits

    for start = 1, 2 do
        local digit = s:sub(start, start)
        local all_same = true
        for i = start, start + 6 do
            if s:sub(i, i) ~= digit then
                all_same = false
                break
            end
        end

        if all_same then
            -- Make sure it's not 8 in a row
            local is_eight = true
            for i = 1, 8 do
                if s:sub(i, i) ~= digit then
                    is_eight = false
                    break
                end
            end

            if not is_eight then
                local positions = {}
                for i = start - 1, start + 5 do
                    table.insert(positions, i)
                end
                return {
                    matched = true,
                    message = "7 consecutive " .. digit .. "s",
                    highlights = {{positions = positions, color = "orange"}}
                }
            end
        end
    end

    return {matched = false}
end
