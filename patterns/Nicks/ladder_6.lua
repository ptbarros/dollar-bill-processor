--[[
Pattern: NICKS_LADDER_6
DisplayName: 6 Digit Ladder
Description: Exactly 6 consecutive digits in strictly ascending or descending order
Tier: 3
Examples: ["12345600", "00123456", "01234500", "65432100", "00654321", "09876500"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Helper to check if a segment is an ascending ladder
    local function is_ascending(seg)
        for i = 1, #seg - 1 do
            local curr = tonumber(seg:sub(i, i))
            local next_d = tonumber(seg:sub(i + 1, i + 1))
            if next_d - curr ~= 1 then return false end
        end
        return true
    end

    -- Helper to check if a segment is a descending ladder
    local function is_descending(seg)
        for i = 1, #seg - 1 do
            local curr = tonumber(seg:sub(i, i))
            local next_d = tonumber(seg:sub(i + 1, i + 1))
            if curr - next_d ~= 1 then return false end
        end
        return true
    end

    -- Check each starting position for a 6-digit ladder
    for start = 1, 3 do
        local segment = s:sub(start, start + 5)
        local is_asc = is_ascending(segment)
        local is_desc = is_descending(segment)

        if is_asc or is_desc then
            -- Check it's not part of a longer ladder
            local extends_before = false
            local extends_after = false

            if start > 1 then
                local prev_char = tonumber(s:sub(start - 1, start - 1))
                local first_char = tonumber(segment:sub(1, 1))
                if is_asc and first_char - prev_char == 1 then extends_before = true end
                if is_desc and prev_char - first_char == 1 then extends_before = true end
            end

            if start + 5 < 8 then
                local next_char = tonumber(s:sub(start + 6, start + 6))
                local last_char = tonumber(segment:sub(6, 6))
                if is_asc and next_char - last_char == 1 then extends_after = true end
                if is_desc and last_char - next_char == 1 then extends_after = true end
            end

            if not extends_before and not extends_after then
                local positions = {}
                for i = start - 1, start + 4 do
                    table.insert(positions, i)
                end
                local direction = is_asc and "ascending" or "descending"
                local color = is_asc and "lime" or "cyan"
                return {
                    matched = true,
                    message = "6-digit " .. direction .. " ladder",
                    highlights = {{positions = positions, color = color}}
                }
            end
        end
    end

    return {matched = false}
end
