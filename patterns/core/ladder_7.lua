--[[
Pattern: NICKS_LADDER_7
DisplayName: 7 Digit Ladder
Catalog: C-039
Description: Exactly 7 consecutive digits in strictly ascending or descending order
Tier: 3
Odds: 1 in 705,882 (136 per 96M)
Examples: ["12345670", "02345678", "98765430"]
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
                -- Group box + arrow matching 4 Digit Ladder (Ed review): ascending
                -- points left->right.
                return {
                    matched = true,
                    message = "7-digit ascending ladder",
                    highlights = {},
                    group_boxes = {{from = positions[1], to = positions[#positions], color = "blue", thickness = 3}},
                    connectors = {{from = positions[1], to = positions[#positions], color = "orange", style = "arrow"}}
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
                -- Group box + arrow matching 4 Digit Ladder (Ed review): descending
                -- points right->left.
                return {
                    matched = true,
                    message = "7-digit descending ladder",
                    highlights = {},
                    group_boxes = {{from = positions[1], to = positions[#positions], color = "blue", thickness = 3}},
                    connectors = {{from = positions[#positions], to = positions[1], color = "orange", style = "arrow"}}
                }
            end
        end
    end

    return {matched = false}
end
