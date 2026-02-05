--[[
Pattern: NICKS_LADDER_7
DisplayName: 7 Digit Ladder
Description: Exactly 7 consecutive digits in strictly ascending or descending order
Tier: 2
Examples: ["01234567", "12345670", "02345678", "87654321", "98765430", "09876543"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Check each starting position for a 7-digit ladder
    for start = 1, 2 do
        local segment = s:sub(start, start + 6)

        -- Check ascending
        local is_asc = true
        for i = 1, 6 do
            local curr = tonumber(segment:sub(i, i))
            local next_d = tonumber(segment:sub(i + 1, i + 1))
            if next_d - curr ~= 1 then
                is_asc = false
                break
            end
        end

        if is_asc then
            -- Check it's not part of 8-digit ladder
            local is_8_ladder = false
            if start == 1 then
                local last = tonumber(s:sub(8, 8))
                local prev = tonumber(segment:sub(7, 7))
                if last - prev == 1 then is_8_ladder = true end
            else
                local first = tonumber(s:sub(1, 1))
                local second = tonumber(segment:sub(1, 1))
                if second - first == 1 then is_8_ladder = true end
            end

            if not is_8_ladder then
                local positions = {}
                for i = start - 1, start + 5 do
                    table.insert(positions, i)
                end
                return {
                    matched = true,
                    message = "7-digit ascending ladder",
                    highlights = {{positions = positions, color = "lime"}}
                }
            end
        end

        -- Check descending
        local is_desc = true
        for i = 1, 6 do
            local curr = tonumber(segment:sub(i, i))
            local next_d = tonumber(segment:sub(i + 1, i + 1))
            if curr - next_d ~= 1 then
                is_desc = false
                break
            end
        end

        if is_desc then
            -- Check it's not part of 8-digit ladder
            local is_8_ladder = false
            if start == 1 then
                local last = tonumber(s:sub(8, 8))
                local prev = tonumber(segment:sub(7, 7))
                if prev - last == 1 then is_8_ladder = true end
            else
                local first = tonumber(s:sub(1, 1))
                local second = tonumber(segment:sub(1, 1))
                if first - second == 1 then is_8_ladder = true end
            end

            if not is_8_ladder then
                local positions = {}
                for i = start - 1, start + 5 do
                    table.insert(positions, i)
                end
                return {
                    matched = true,
                    message = "7-digit descending ladder",
                    highlights = {{positions = positions, color = "cyan"}}
                }
            end
        end
    end

    return {matched = false}
end
