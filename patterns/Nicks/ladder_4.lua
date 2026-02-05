--[[
Pattern: NICKS_LADDER_4
DisplayName: 4 Digit Ladder
Description: Exactly 4 consecutive digits in strictly ascending or descending order
Tier: 5
Examples: ["12340000", "00001234", "00123400", "43210000", "00004321", "00987600"]
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

    for start = 1, 5 do
        local segment = s:sub(start, start + 3)
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

            if start + 3 < 8 then
                local next_char = tonumber(s:sub(start + 4, start + 4))
                local last_char = tonumber(segment:sub(4, 4))
                if is_asc and next_char - last_char == 1 then extends_after = true end
                if is_desc and last_char - next_char == 1 then extends_after = true end
            end

            if not extends_before and not extends_after then
                local positions = {}
                for i = start - 1, start + 2 do
                    table.insert(positions, i)
                end
                local direction = is_asc and "ascending" or "descending"
                local color = is_asc and "lime" or "cyan"
                return {
                    matched = true,
                    message = "4-digit " .. direction .. " ladder",
                    highlights = {{positions = positions, color = color}}
                }
            end
        end
    end

    return {matched = false}
end
