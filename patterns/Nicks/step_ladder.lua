--[[
Pattern: NICKS_STEP_LADDER
DisplayName: Step Ladder
Description: First 4 digits sorted, last 4 digits sorted, with step of 1 between
Tier: 4
Examples: ["12345678", "01234567", "87654321", "98765432"]
--]]

function match(ctx)
    local s = ctx.digits
    local first_four = s:sub(1, 4)
    local last_four = s:sub(5, 8)

    -- Check ascending step ladder
    local first_asc = true
    local last_asc = true
    for i = 1, 3 do
        local c1 = tonumber(first_four:sub(i, i))
        local n1 = tonumber(first_four:sub(i + 1, i + 1))
        local c2 = tonumber(last_four:sub(i, i))
        local n2 = tonumber(last_four:sub(i + 1, i + 1))
        if n1 - c1 ~= 1 then first_asc = false end
        if n2 - c2 ~= 1 then last_asc = false end
    end

    if first_asc and last_asc then
        local last_of_first = tonumber(first_four:sub(4, 4))
        local first_of_last = tonumber(last_four:sub(1, 1))
        if first_of_last - last_of_first == 1 then
            return {
                matched = true,
                message = "Ascending step ladder",
                highlights = {
                    {positions = {0,1,2,3}, color = "lime"},
                    {positions = {4,5,6,7}, color = "cyan"}
                },
                connectors = {{from = 3, to = 4, color = "gold", style = "arc"}}
            }
        end
    end

    -- Check descending step ladder
    local first_desc = true
    local last_desc = true
    for i = 1, 3 do
        local c1 = tonumber(first_four:sub(i, i))
        local n1 = tonumber(first_four:sub(i + 1, i + 1))
        local c2 = tonumber(last_four:sub(i, i))
        local n2 = tonumber(last_four:sub(i + 1, i + 1))
        if c1 - n1 ~= 1 then first_desc = false end
        if c2 - n2 ~= 1 then last_desc = false end
    end

    if first_desc and last_desc then
        local last_of_first = tonumber(first_four:sub(4, 4))
        local first_of_last = tonumber(last_four:sub(1, 1))
        if last_of_first - first_of_last == 1 then
            return {
                matched = true,
                message = "Descending step ladder",
                highlights = {
                    {positions = {0,1,2,3}, color = "cyan"},
                    {positions = {4,5,6,7}, color = "lime"}
                },
                connectors = {{from = 3, to = 4, color = "gold", style = "arc"}}
            }
        end
    end

    return {matched = false}
end
