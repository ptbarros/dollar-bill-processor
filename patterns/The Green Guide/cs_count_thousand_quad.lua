--[[
Pattern: CS_COUNT_THOUSAND_QUAD
DisplayName: CS-Count Thousand Quad
Description: Two 4-digit halves where only the first digit differs by 1. The last three digits of each half are identical. e.g., M 1234 2234 M or M 2234 1234 M.
BookRef: CS-870
Tier: 3
Examples: ["12342234", "22341234"]
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Last 3 digits of each half must match: positions 1-3 == 5-7
    if d:sub(2, 4) ~= d:sub(6, 8) then return {matched = false} end

    -- First digit of each half (positions 0 and 4) must differ by ±1
    local a = tonumber(d:sub(1, 1))
    local b = tonumber(d:sub(5, 5))
    local diff = b - a
    if diff ~= 1 and diff ~= -1 then return {matched = false} end

    local direction = diff == 1 and "up" or "down"
    return {
        matched = true,
        highlights = {
            {positions = {0, 4}, color = "gold"}
        },
        group_boxes = {
            {from = 0, to = 3, color = "cyan", thickness = 2},
            {from = 4, to = 7, color = "cyan", thickness = 2},
        },
        connectors = {
            {from = 0, to = 4, color = "lime", style = "arc"},
        },
        message = "Thousands digit counts " .. direction .. ": " .. d:sub(1,4) .. " → " .. d:sub(5,8) .. " (CS-870)"
    }
end
