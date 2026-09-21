--[[
Pattern: CS_SPLIT_COUNT
DisplayName: CS-Split Count
Description: A single matching digit caps both ends around two side-by-side triples that step up or down by one (e.g. 2·000·111·2).
BookRef: CS-890
Tier: 3
Examples: ["20001112", "29998882"]
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Bookend: position 0 == position 7
    if d:sub(1, 1) ~= d:sub(8, 8) then return {matched = false} end

    -- First triple: positions 1-3 all same digit
    local x = d:sub(2, 2)
    if d:sub(3, 3) ~= x or d:sub(4, 4) ~= x then return {matched = false} end

    -- Second triple: positions 4-6 all same digit
    local y = d:sub(5, 5)
    if d:sub(6, 6) ~= y or d:sub(7, 7) ~= y then return {matched = false} end

    -- Triples must differ by exactly ±1
    local diff = tonumber(y) - tonumber(x)
    if diff ~= 1 and diff ~= -1 then return {matched = false} end

    local outer = d:sub(1, 1)
    local direction = diff == 1 and "up" or "down"
    return {
        matched = true,
        highlights = {
            {positions = {0, 7}, color = "cyan"}
        },
        group_boxes = {
            {from = 1, to = 3, color = "gold", thickness = 3},
            {from = 4, to = 6, color = "orange", thickness = 3},
        },
        connectors = {
            {from = 0, to = 7, color = "cyan", style = "arc"},
            {from = 3, to = 4, color = "lime", style = "line"},
        },
        message = outer .. " bookends " .. x .. x .. x .. " → " .. y .. y .. y .. " counts " .. direction .. " (CS-890)"
    }
end
