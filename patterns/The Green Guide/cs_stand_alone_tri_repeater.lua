--[[
Pattern: CS_STAND_ALONE_TRI_REPEATER
DisplayName: CS-Stand Alone Tri Repeater
Description: Three CS-20AKs repeating within zeros (ABCABC at positions 2-7, surrounded by zeros). e.g., M 0301301 0 M or M 0123123 0 M.
BookRef: CS-1720
Tier: 5
Examples: ["03013010", "01231230"]
Odds: 1 in 900
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Positions 1 and 8 must be zero
    if d:sub(1, 1) ~= "0" or d:sub(8, 8) ~= "0" then
        return {matched = false}
    end

    -- ABCABC must be at positions 2-7 (Lua 2-7)
    -- d[2]==d[5], d[3]==d[6], d[4]==d[7]
    local a = d:sub(2, 2)
    local b = d:sub(3, 3)
    local c = d:sub(4, 4)
    local a2 = d:sub(5, 5)
    local b2 = d:sub(6, 6)
    local c2 = d:sub(7, 7)

    if a ~= a2 or b ~= b2 or c ~= c2 then
        return {matched = false}
    end

    -- At least 2 distinct digits (not all same)
    if unique_count(d:sub(2, 7)) < 2 then
        return {matched = false}
    end

    -- The motif digits must be non-zero
    if a == "0" and b == "0" and c == "0" then
        return {matched = false}
    end

    return {
        matched = true,
        group_boxes = {
            {from = 1, to = 3, color = "orange", thickness = 2},
            {from = 4, to = 6, color = "orange", thickness = 2}
        },
        connectors = {
            {from = 1, to = 4, color = "orange", style = "arc"},
            {from = 2, to = 5, color = "coral", style = "arc"},
            {from = 3, to = 6, color = "cyan", style = "arc"}
        },
        message = a .. b .. c .. " tri-repeater stand-alone (CS-Stand Alone Tri Repeater)"
    }
end
