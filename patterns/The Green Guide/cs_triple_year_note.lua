--[[
Pattern: CS_TRIPLE_YEAR_NOTE
DisplayName: CS-Triple Year Note
Description: Three overlapping valid years (default 1700-2099, editable below) starting at different positions among positions 1-5 in the serial.
BookRef: CS-720
Tier: 6
Examples: ["19192012", "19201920", "11992012"]
--]]

function match(ctx)
    -- === Editable year range (inclusive) ===
    local YEAR_MIN = 1700   -- earliest year to accept
    local YEAR_MAX = 2099   -- latest year to accept
    -- =======================================

    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Find all positions (1-5) that start a valid year
    local valid_positions = {}
    for start = 1, 5 do
        local year_str = d:sub(start, start + 3)
        local year = tonumber(year_str)
        if year and (year >= YEAR_MIN and year <= YEAR_MAX) then
            table.insert(valid_positions, {start = start, year = year_str})
        end
    end

    if #valid_positions >= 3 then
        local boxes = {}
        local colors = {"cyan", "lime", "orange", "coral", "gold"}
        local msg_parts = {}
        for i, vp in ipairs(valid_positions) do
            local s0 = vp.start - 1
            table.insert(boxes, {from = s0, to = s0 + 3, color = colors[((i - 1) % #colors) + 1], thickness = 2})
            table.insert(msg_parts, vp.year)
        end
        return {
            matched = true,
            group_boxes = boxes,
            message = "Triple Year Note: " .. table.concat(msg_parts, ", ")
        }
    end

    return {matched = false}
end
