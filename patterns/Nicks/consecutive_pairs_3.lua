--[[
Pattern: CONSECUTIVE_PAIRS_3
DisplayName: 3 Consecutive Pairs
Description: 3 consecutive pairs within the serial
Tier: 5
Examples: ["11223312", "12233445", "00112234", "99887700"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Look for 3 consecutive pairs (6 digits)
    for start = 1, 3 do
        local valid = true
        for i = start, start + 4, 2 do
            local pair = s:sub(i, i + 1)
            if pair:sub(1, 1) ~= pair:sub(2, 2) then
                valid = false
                break
            end
            -- Make sure next pair is different
            if i + 2 <= start + 4 then
                local next_pair = s:sub(i + 2, i + 3)
                if pair:sub(1, 1) == next_pair:sub(1, 1) then
                    valid = false
                    break
                end
            end
        end

        if valid then
            local positions = {}
            for i = start - 1, start + 4 do
                table.insert(positions, i)
            end
            return {
                matched = true,
                message = "3 consecutive pairs",
                highlights = {{positions = positions, color = "orange"}}
            }
        end
    end

    return {matched = false}
end
