--[[
Pattern: NICKS_CHUNKY_LADDER_3
DisplayName: 3 Digit Chunky Ladder
Description: 3 unique digits all in sorted order (ascending or descending)
Tier: 7
Examples: ["00000123", "01222222", "98700000", "33332100"]
--]]

function match(ctx)
    local s = ctx.digits

    local unique = {}
    for i = 1, 8 do
        unique[s:sub(i, i)] = true
    end

    local count = 0
    for _ in pairs(unique) do count = count + 1 end

    if count ~= 3 then
        return {matched = false}
    end

    local sorted_asc = {}
    for i = 1, 8 do
        table.insert(sorted_asc, s:sub(i, i))
    end
    table.sort(sorted_asc)

    if table.concat(sorted_asc) == s then
        return {
            matched = true,
            message = "3-digit chunky ladder (ascending)",
            highlights = {{positions = {0,1,2,3,4,5,6,7}, color = "lime"}}
        }
    end

    table.sort(sorted_asc, function(a, b) return a > b end)
    if table.concat(sorted_asc) == s then
        return {
            matched = true,
            message = "3-digit chunky ladder (descending)",
            highlights = {{positions = {0,1,2,3,4,5,6,7}, color = "cyan"}}
        }
    end

    return {matched = false}
end
