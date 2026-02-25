--[[
Pattern: CS_DOUBLE_QUADS_COUNT
DisplayName: CS-Double Quads Count Note
Description: Two CS-Quads (each half all same digit) that increase or decrease by 1. e.g., M 6666 7777 M or M 4444 3333 M.
BookRef: CS-880
Tier: 2
Examples: ["66667777", "44443333", "11112222"]
Price: $50-$300
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- First quad: positions 0-3 all same digit
    local a = d:sub(1, 1)
    if d:sub(2, 2) ~= a or d:sub(3, 3) ~= a or d:sub(4, 4) ~= a then
        return {matched = false}
    end

    -- Second quad: positions 4-7 all same digit
    local b = d:sub(5, 5)
    if d:sub(6, 6) ~= b or d:sub(7, 7) ~= b or d:sub(8, 8) ~= b then
        return {matched = false}
    end

    -- Must differ by exactly ±1
    local diff = tonumber(b) - tonumber(a)
    if diff ~= 1 and diff ~= -1 then return {matched = false} end

    local direction = diff == 1 and "up" or "down"
    return {
        matched = true,
        group_boxes = {
            {from = 0, to = 3, color = "gold", thickness = 3},
            {from = 4, to = 7, color = "orange", thickness = 3},
        },
        connectors = {
            {from = 3, to = 4, color = "lime", style = "line"},
        },
        message = a .. a .. a .. a .. " → " .. b .. b .. b .. b .. " counts " .. direction .. " (CS-880)"
    }
end
