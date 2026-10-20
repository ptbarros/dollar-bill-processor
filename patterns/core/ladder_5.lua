--[[
Pattern: NICKS_LADDER_5
DisplayName: 5 Digit Ladder
Catalog: C-099
Description: Exactly 5 consecutive digits in strictly ascending or descending order
Tier: 6
Odds: 1 in 2,391 (40,146 per 96M)
Examples: ["12345000", "00054321", "00987650"]
--]]

function match(ctx)
    local s = ctx.digits

    local function is_ascending(seg)
        for i = 1, #seg - 1 do
            local curr = tonumber(seg:sub(i, i))
            local next_d = tonumber(seg:sub(i + 1, i + 1))
            if next_d - curr ~= 1 then return false end
        end
        return true
    end

    local function is_descending(seg)
        for i = 1, #seg - 1 do
            local curr = tonumber(seg:sub(i, i))
            local next_d = tonumber(seg:sub(i + 1, i + 1))
            if curr - next_d ~= 1 then return false end
        end
        return true
    end

    for start = 1, 4 do
        local segment = s:sub(start, start + 4)
        local is_asc = is_ascending(segment)
        local is_desc = is_descending(segment)

        if is_asc or is_desc then
            local extends_before = false
            local extends_after = false

            if start > 1 then
                local prev_char = tonumber(s:sub(start - 1, start - 1))
                local first_char = tonumber(segment:sub(1, 1))
                if is_asc and first_char - prev_char == 1 then extends_before = true end
                if is_desc and prev_char - first_char == 1 then extends_before = true end
            end

            if start + 4 < 8 then
                local next_char = tonumber(s:sub(start + 5, start + 5))
                local last_char = tonumber(segment:sub(5, 5))
                if is_asc and next_char - last_char == 1 then extends_after = true end
                if is_desc and last_char - next_char == 1 then extends_after = true end
            end

            if not extends_before and not extends_after then
                local positions = {}
                for i = start - 1, start + 3 do
                    table.insert(positions, i)
                end
                local direction = is_asc and "ascending" or "descending"
                local from_pos = positions[1]
                local to_pos = positions[#positions]
                -- Group box + an arrow pointing the way the ladder climbs, matching
                -- 4 Digit Ladder (Ed review): left->right ascending, right->left descending.
                local a_from, a_to
                if is_asc then a_from, a_to = from_pos, to_pos
                else a_from, a_to = to_pos, from_pos end
                return {
                    matched = true,
                    message = "5-digit " .. direction .. " ladder",
                    highlights = {},
                    group_boxes = {{from = from_pos, to = to_pos, color = "blue", thickness = 3}},
                    connectors = {{from = a_from, to = a_to, color = "orange", style = "arrow"}}
                }
            end
        end
    end

    return {matched = false}
end
