--[[
Pattern: NICKS_CHUNKY_LADDER_7
DisplayName: 7 Digit Chunky Ladder
Description: 7 unique digits all in sorted order (ascending or descending)
Tier: 3
Examples: ["01234567", "12345678", "23456789", "98765432", "87654321"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Count unique digits
    local unique = {}
    for i = 1, 8 do
        unique[s:sub(i, i)] = true
    end

    local count = 0
    for _ in pairs(unique) do count = count + 1 end

    if count ~= 7 then
        return {matched = false}
    end

    -- Check if sorted ascending
    local sorted_asc = {}
    for i = 1, 8 do
        table.insert(sorted_asc, s:sub(i, i))
    end
    table.sort(sorted_asc)

    if table.concat(sorted_asc) == s then
        return {
            matched = true,
            message = "7-digit chunky ladder (ascending)",
            highlights = {{positions = {0,1,2,3,4,5,6,7}, color = "lime"}}
        }
    end

    -- Check if sorted descending
    table.sort(sorted_asc, function(a, b) return a > b end)
    if table.concat(sorted_asc) == s then
        return {
            matched = true,
            message = "7-digit chunky ladder (descending)",
            highlights = {{positions = {0,1,2,3,4,5,6,7}, color = "cyan"}}
        }
    end

    return {matched = false}
end
