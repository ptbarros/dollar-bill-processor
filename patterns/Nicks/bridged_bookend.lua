--[[
Pattern: NICKS_BRIDGED_BOOKEND
DisplayName: Bridged Bookend
Description: First and last digit match, 2-4 and 5-7 are identical groups (ABBBCCCA)
Tier: 4
Examples: ["12223331", "01112220", "98887779", "15556661"]
--]]

function match(ctx)
    local s = ctx.digits

    -- First and last must match
    if s:sub(1, 1) ~= s:sub(8, 8) then
        return {matched = false}
    end

    -- Positions 2-4 must be identical
    local group1 = s:sub(2, 2)
    if s:sub(3, 3) ~= group1 or s:sub(4, 4) ~= group1 then
        return {matched = false}
    end

    -- Positions 5-7 must be identical
    local group2 = s:sub(5, 5)
    if s:sub(6, 6) ~= group2 or s:sub(7, 7) ~= group2 then
        return {matched = false}
    end

    -- Groups must be different
    if group1 == group2 then
        return {matched = false}
    end

    return {
        matched = true,
        message = "Bridged bookend: " .. s:sub(1, 1) .. " + " .. group1 .. "×3 + " .. group2 .. "×3 + " .. s:sub(8, 8),
        highlights = {
            {positions = {0, 7}, color = "gold"}
        },
        group_boxes = {
            {from = 1, to = 3, color = "orange"},
            {from = 4, to = 6, color = "cyan"}
        },
        connectors = {{from = 0, to = 7, color = "gold", style = "arc"}}
    }
end
