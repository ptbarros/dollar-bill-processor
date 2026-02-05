--[[
Pattern: NICKS_BINARY_REPEATER
DisplayName: Binary Repeater
Description: Repeater with only 2 unique digits
Tier: 3
Examples: ["12121212", "00110011", "98989898", "12001200"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Check repeater
    local first_half = s:sub(1, 4)
    local second_half = s:sub(5, 8)

    if first_half ~= second_half then
        return {matched = false}
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
            message = "Binary repeater",
            group_boxes = {
                {from = 0, to = 3, color = "cyan"},
                {from = 4, to = 7, color = "cyan"}
            }
        }
    end

    return {matched = false}
end
