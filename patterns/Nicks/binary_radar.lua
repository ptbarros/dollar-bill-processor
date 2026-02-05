--[[
Pattern: NICKS_BINARY_RADAR
DisplayName: Binary Radar
Description: Palindrome with only 2 unique digits
Tier: 3
Examples: ["12211221", "00111100", "98899889", "12000021"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Check palindrome
    for i = 1, 4 do
        if s:sub(i, i) ~= s:sub(9 - i, 9 - i) then
            return {matched = false}
        end
    end

    -- Check binary (2 unique)
    local unique = {}
    for i = 1, 8 do
        unique[s:sub(i, i)] = true
    end

    local count = 0
    for _ in pairs(unique) do count = count + 1 end

    if count == 2 then
        return {
            matched = true,
            message = "Binary radar",
            highlights = {{positions = {0,1,2,3,4,5,6,7}, color = "cyan"}},
            connectors = {
                {from = 0, to = 7, color = "cyan", style = "arc"},
                {from = 1, to = 6, color = "cyan", style = "arc"},
                {from = 2, to = 5, color = "cyan", style = "arc"},
                {from = 3, to = 4, color = "cyan", style = "arc"}
            }
        }
    end

    return {matched = false}
end
