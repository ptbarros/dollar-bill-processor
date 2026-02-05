--[[
Pattern: LADDER_8
DisplayName: 8 Digit Ladder
Description: All 8 digits in strictly ascending or descending order (01234567 or 98765432)
Tier: 1
Examples: ["01234567", "12345678", "23456789", "98765432", "87654321", "76543210"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Check ascending
    local is_asc = true
    for i = 1, 7 do
        local curr = tonumber(s:sub(i, i))
        local next = tonumber(s:sub(i + 1, i + 1))
        if next - curr ~= 1 then
            is_asc = false
            break
        end
    end

    if is_asc then
        return {
            matched = true,
            message = "Full 8-digit ascending ladder",
            highlights = {{positions = {0,1,2,3,4,5,6,7}, color = "lime"}}
        }
    end

    -- Check descending
    local is_desc = true
    for i = 1, 7 do
        local curr = tonumber(s:sub(i, i))
        local next = tonumber(s:sub(i + 1, i + 1))
        if curr - next ~= 1 then
            is_desc = false
            break
        end
    end

    if is_desc then
        return {
            matched = true,
            message = "Full 8-digit descending ladder",
            highlights = {{positions = {0,1,2,3,4,5,6,7}, color = "cyan"}}
        }
    end

    return {matched = false}
end
