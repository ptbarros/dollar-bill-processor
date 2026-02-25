--[[
Pattern: CS_COUNT_TEN_QUADS
DisplayName: CS-Count Ten Quads
Description: Two 4-digit halves where only the second-to-last digit differs by 1. The other three digits of each half are identical. e.g., M 1234 1244 M.
BookRef: CS-850
Tier: 3
Examples: ["12341244", "12441234", "11711181"]
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Matching positions: 0==4, 1==5, 3==7
    if d:sub(1, 1) ~= d:sub(5, 5) then return {matched = false} end
    if d:sub(2, 2) ~= d:sub(6, 6) then return {matched = false} end
    if d:sub(4, 4) ~= d:sub(8, 8) then return {matched = false} end

    -- Differing position: 2 and 6 (0-indexed) must differ by ±1
    local a = tonumber(d:sub(3, 3))
    local b = tonumber(d:sub(7, 7))
    local diff = b - a
    if diff ~= 1 and diff ~= -1 then return {matched = false} end

    local direction = diff == 1 and "up" or "down"
    return {
        matched = true,
        highlights = {
            {positions = {2, 6}, color = "gold"}
        },
        group_boxes = {
            {from = 0, to = 3, color = "cyan", thickness = 2},
            {from = 4, to = 7, color = "cyan", thickness = 2},
        },
        connectors = {
            {from = 2, to = 6, color = "lime", style = "arc"},
        },
        message = "Tens digit counts " .. direction .. ": " .. d:sub(1,4) .. " → " .. d:sub(5,8) .. " (CS-850)"
    }
end
