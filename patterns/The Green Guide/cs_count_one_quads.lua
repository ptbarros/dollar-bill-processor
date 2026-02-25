--[[
Pattern: CS_COUNT_ONE_QUADS
DisplayName: CS-Count One Quads
Description: Two 4-digit halves where only the last digit differs by 1. The first three digits of each half are identical. e.g., M 1234 1235 M or M 1235 1234 M.
BookRef: CS-840
Tier: 3
Examples: ["12341235", "12351234", "11151116"]
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- First 3 digits of each half must match: positions 0-2 == 4-6
    if d:sub(1, 3) ~= d:sub(5, 7) then return {matched = false} end

    -- Last digit of each half (positions 3 and 7) must differ by ±1
    local a = tonumber(d:sub(4, 4))
    local b = tonumber(d:sub(8, 8))
    local diff = b - a
    if diff ~= 1 and diff ~= -1 then return {matched = false} end

    local direction = diff == 1 and "up" or "down"
    return {
        matched = true,
        highlights = {
            {positions = {3, 7}, color = "gold"}
        },
        group_boxes = {
            {from = 0, to = 3, color = "cyan", thickness = 2},
            {from = 4, to = 7, color = "cyan", thickness = 2},
        },
        connectors = {
            {from = 3, to = 7, color = "lime", style = "arc"},
        },
        message = "Last digit counts " .. direction .. ": " .. d:sub(1,4) .. " → " .. d:sub(5,8) .. " (CS-840)"
    }
end
