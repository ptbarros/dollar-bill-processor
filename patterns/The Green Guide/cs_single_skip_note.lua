--[[
Pattern: CS_SINGLE_SKIP_NOTE
DisplayName: CS-Single Skip Note
Description: A CS-40AK that skips every other digit — the same digit at all 4 odd or all 4 even positions. The other 4 positions may not form another CS-40AK. e.g., M 1x1x1x1x M or M x2x2x2x2 M.
BookRef: CS-1590
Tier: 6
Examples: ["10101012", "01010102", "40404041"]
Odds: 1 in 131,220
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Check odd positions (1,3,5,7 → Lua 1-indexed: 1,3,5,7)
    local function check_skip(pos1, pos2, pos3, pos4, other1, other2, other3, other4)
        local skip_digit = d:sub(pos1, pos1)
        if d:sub(pos2, pos2) ~= skip_digit then return nil end
        if d:sub(pos3, pos3) ~= skip_digit then return nil end
        if d:sub(pos4, pos4) ~= skip_digit then return nil end

        -- The other 4 positions must NOT all be the same digit (that would be Super Repeater)
        local other_digit = d:sub(other1, other1)
        if d:sub(other2, other2) == other_digit and
           d:sub(other3, other3) == other_digit and
           d:sub(other4, other4) == other_digit then
            return nil  -- Super Repeater (CS-1530/CS-1600) — skip
        end

        return skip_digit
    end

    -- Odd positions: 1,3,5,7 (Lua 1-indexed)
    local skip_d = check_skip(1, 3, 5, 7, 2, 4, 6, 8)
    if skip_d then
        return {
            matched = true,
            highlights = {
                {positions = {0, 2, 4, 6}, color = "gold"},
            },
            message = skip_d .. " at every odd position (CS-Single Skip Note)"
        }
    end

    -- Even positions: 2,4,6,8 (Lua 1-indexed)
    skip_d = check_skip(2, 4, 6, 8, 1, 3, 5, 7)
    if skip_d then
        return {
            matched = true,
            highlights = {
                {positions = {1, 3, 5, 7}, color = "gold"},
            },
            message = skip_d .. " at every even position (CS-Single Skip Note)"
        }
    end

    return {matched = false}
end
