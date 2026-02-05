--[[
Pattern: NICKS_COUNTING_LADDER
DisplayName: Counting Ladder
Description: Pattern X0X1X2X3 where odd positions count up/down
Tier: 4
Examples: ["10203040", "20314050", "90807060", "50403020"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Extract odd positions (1,3,5,7 = indices 1,3,5,7)
    local first_digits = s:sub(1, 1) .. s:sub(3, 3) .. s:sub(5, 5) .. s:sub(7, 7)
    -- Extract even positions (2,4,6,8 = indices 2,4,6,8)
    local second_digits = s:sub(2, 2) .. s:sub(4, 4) .. s:sub(6, 6) .. s:sub(8, 8)

    -- First digits must all be the same
    if first_digits:sub(1, 1) ~= first_digits:sub(2, 2) or
       first_digits:sub(1, 1) ~= first_digits:sub(3, 3) or
       first_digits:sub(1, 1) ~= first_digits:sub(4, 4) then
        return {matched = false}
    end

    -- Second digits must be ascending or descending by 1
    local is_asc = true
    local is_desc = true
    for i = 1, 3 do
        local curr = tonumber(second_digits:sub(i, i))
        local next = tonumber(second_digits:sub(i + 1, i + 1))
        if next - curr ~= 1 then is_asc = false end
        if curr - next ~= 1 then is_desc = false end
    end

    if not is_asc and not is_desc then
        return {matched = false}
    end

    local base = first_digits:sub(1, 1)
    local direction = is_asc and "ascending" or "descending"

    return {
        matched = true,
        message = "Counting ladder: " .. base .. "X pattern " .. direction,
        highlights = {
            {positions = {0, 2, 4, 6}, color = "orange"},
            {positions = {1, 3, 5, 7}, color = is_asc and "lime" or "cyan"}
        }
    }
end
