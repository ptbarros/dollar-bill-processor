--[[
Pattern: TOMBSTONE
DisplayName: Tombstone
Description: Two 4-digit years, second > first, difference <= 120 years (birth-death)
Tier: 6
Examples: ["19001990", "19202020", "18501970", "19502050"]
--]]

function match(ctx)
    local s = ctx.digits

    local year1 = tonumber(s:sub(1, 4))
    local year2 = tonumber(s:sub(5, 8))

    if not year1 or not year2 then
        return {matched = false}
    end

    -- Both must be plausible years (1200-2050)
    if year1 < 1200 or year1 > 2050 then return {matched = false} end
    if year2 < 1200 or year2 > 2050 then return {matched = false} end

    -- Second must be larger
    if year2 <= year1 then return {matched = false} end

    -- Difference must be <= 120 (reasonable lifespan)
    local diff = year2 - year1
    if diff > 120 then return {matched = false} end

    return {
        matched = true,
        message = "Tombstone: " .. year1 .. "-" .. year2 .. " (" .. diff .. " years)",
        group_boxes = {
            {from = 0, to = 3, color = "gray"},
            {from = 4, to = 7, color = "gray"}
        },
        connectors = {{from = 1, to = 5, color = "gray", style = "line"}}
    }
end
