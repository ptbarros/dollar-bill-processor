--[[
Pattern: CS_COUNT_HUNDRED_QUADS
DisplayName: CS-Count Hundred Quads
Description: Two 4-digit halves where only the third-to-last digit (position 2 in each half) differs by 1. The other three digits are identical. e.g., M 1234 1334 M.
BookRef: CS-860
Tier: 3
Examples: ["12341334", "13341234"]
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Matching positions: 0==4, 2==6, 3==7
    if d:sub(1, 1) ~= d:sub(5, 5) then return {matched = false} end
    if d:sub(3, 3) ~= d:sub(7, 7) then return {matched = false} end
    if d:sub(4, 4) ~= d:sub(8, 8) then return {matched = false} end

    -- Differing position: 1 and 5 (0-indexed) must differ by ±1
    local a = tonumber(d:sub(2, 2))
    local b = tonumber(d:sub(6, 6))
    local diff = b - a
    if diff ~= 1 and diff ~= -1 then return {matched = false} end

    local direction = diff == 1 and "up" or "down"
    return {
        matched = true,
        highlights = {
            {positions = {1, 5}, color = "gold"}
        },
        group_boxes = {
            {from = 0, to = 3, color = "cyan", thickness = 2},
            {from = 4, to = 7, color = "cyan", thickness = 2},
        },
        connectors = {
            {from = 1, to = 5, color = "lime", style = "arc"},
        },
        message = "Hundreds digit counts " .. direction .. ": " .. d:sub(1,4) .. " → " .. d:sub(5,8) .. " (CS-860)"
    }
end
