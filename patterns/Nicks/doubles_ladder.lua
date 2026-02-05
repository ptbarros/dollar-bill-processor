--[[
Pattern: NICKS_DOUBLES_LADDER
DisplayName: Doubles Ladder
Description: AABBCCDD where AA < BB < CC < DD (or descending)
Tier: 4
Examples: ["11223344", "00112233", "99887766", "44332211"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Split into pairs
    local pairs = {}
    for i = 1, 8, 2 do
        local pair = s:sub(i, i + 1)
        -- Each pair must have identical digits
        if pair:sub(1, 1) ~= pair:sub(2, 2) then
            return {matched = false}
        end
        table.insert(pairs, tonumber(pair:sub(1, 1)))
    end

    -- Check ascending
    local is_asc = true
    for i = 1, 3 do
        if pairs[i + 1] <= pairs[i] then
            is_asc = false
            break
        end
    end

    if is_asc then
        return {
            matched = true,
            message = "Doubles ladder (ascending)",
            group_boxes = {
                {from = 0, to = 1, color = "lime"},
                {from = 2, to = 3, color = "lime"},
                {from = 4, to = 5, color = "lime"},
                {from = 6, to = 7, color = "lime"}
            }
        }
    end

    -- Check descending
    local is_desc = true
    for i = 1, 3 do
        if pairs[i + 1] >= pairs[i] then
            is_desc = false
            break
        end
    end

    if is_desc then
        return {
            matched = true,
            message = "Doubles ladder (descending)",
            group_boxes = {
                {from = 0, to = 1, color = "cyan"},
                {from = 2, to = 3, color = "cyan"},
                {from = 4, to = 5, color = "cyan"},
                {from = 6, to = 7, color = "cyan"}
            }
        }
    end

    return {matched = false}
end
