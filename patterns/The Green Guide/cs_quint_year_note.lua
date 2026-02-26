--[[
Pattern: CS_QUINT_YEAR_NOTE
DisplayName: CS-Quint Year Note
Description: All five possible positions (1-5) in the serial each start a valid year (1700-2099).
BookRef: CS-740
Tier: 4
Examples: ["11112010"]
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    local valid_positions = {}
    for start = 1, 5 do
        local year_str = d:sub(start, start + 3)
        local year = tonumber(year_str)
        if year and is_valid_year(year) then
            table.insert(valid_positions, {start = start, year = year_str})
        else
            return {matched = false}
        end
    end

    local boxes = {}
    local colors = {"cyan", "lime", "orange", "coral", "gold"}
    local msg_parts = {}
    for i, vp in ipairs(valid_positions) do
        local s0 = vp.start - 1
        table.insert(boxes, {from = s0, to = s0 + 3, color = colors[i], thickness = 2})
        table.insert(msg_parts, vp.year)
    end
    return {
        matched = true,
        group_boxes = boxes,
        message = "Quint Year Note: " .. table.concat(msg_parts, ", ")
    }
end
