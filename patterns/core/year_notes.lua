--[[
Pattern: YEAR_NOTES
DisplayName: Year Note
Catalog: C-136
Description: Four digits in a row, anywhere in the serial, read as a number from 1700 to 2099 — a year, real or future. A serial holding two separate years (e.g. 2091 and 1890) has both marked.
Tier: 10
AKA: Birth Year, Year Serial, Year in Serial
Odds: 1 in 5 (18,572,800 per 96M)
Examples: ["19751234", "12197534", "12341975"]
--]]

function match(ctx)
    -- === Editable year range (inclusive) ===
    local YEAR_MIN = 1700   -- earliest year to accept
    local YEAR_MAX = 2099   -- latest year to accept
    -- =======================================

    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Mark every year, scanning left to right. After a year is found, skip
    -- past it so boxes never overlap (20911890 -> 2091 and 1890).
    local colors = {"blue", "orange"}
    local boxes, years = {}, {}
    local start = 1
    while start <= 5 do
        local year_str = d:sub(start, start + 3)
        local year = tonumber(year_str)
        if year and year >= YEAR_MIN and year <= YEAR_MAX then
            local s0 = start - 1
            table.insert(boxes, {from = s0, to = s0 + 3,
                                 color = colors[#boxes + 1], thickness = 3})
            table.insert(years, year_str)
            start = start + 4
        else
            start = start + 1
        end
    end

    if #boxes == 0 then return {matched = false} end
    return {
        matched = true,
        group_boxes = boxes,
        message = "Year Note: " .. table.concat(years, " and ")
    }
end
