--[[
Pattern: DOUBLE_YEAR
DisplayName: CS-Double Year Note
Description: Two consecutive valid years (1940-2030) forming the full 8-digit serial.
BookRef: CS-710
Tier: 6
Examples: ["19741975", "20002001", "19992000"]
--]]

function match(ctx)
    local d = ctx.digits
    
    -- Extract first year (positions 0-3) and second year (positions 4-7)
    local year1_str = d:sub(1, 4)
    local year2_str = d:sub(5, 8)
    
    local year1 = tonumber(year1_str)
    local year2 = tonumber(year2_str)
    
    -- Check if both are valid years in range 1940-2030
    if not year1 or not year2 then
        return {matched = false}
    end
    
    if year1 < 1940 or year1 > 2030 then
        return {matched = false}
    end
    
    if year2 < 1940 or year2 > 2030 then
        return {matched = false}
    end
    
    -- Match! Highlight both years with group boxes
    return {
        matched = true,
        group_boxes = {
            {from = 0, to = 3, color = "cyan", thickness = 2},
            {from = 4, to = 7, color = "lime", thickness = 2}
        },
        connectors = {
            {from = 1, to = 5, color = "orange", style = "arc"}
        },
        message = year1_str .. " + " .. year2_str
    }
end
