--[[
Pattern: DUPLICATE_SN
DisplayName: Duplicate S/N
Description: B* star note in specific ranges (potential duplicate serial)
Tier: 3
Examples: ["00000001", "00100000", "03200001", "05000000"]
--]]

function match(ctx)
    local full = ctx.full_serial
    local s = ctx.digits

    -- Must be B prefix and * suffix
    if full:sub(1, 1) ~= "B" or full:sub(-1) ~= "*" then
        return {matched = false}
    end

    -- Check ranges: 00000001-00250000 or 03200001-09600000
    local num = tonumber(s)
    if num then
        if (num >= 1 and num <= 250000) or (num >= 3200001 and num <= 9600000) then
            return {
                matched = true,
                message = "Potential duplicate serial (B* range)",
                highlights = {{positions = {0,1,2,3,4,5,6,7}, color = "red"}}
            }
        end
    end

    return {matched = false}
end
