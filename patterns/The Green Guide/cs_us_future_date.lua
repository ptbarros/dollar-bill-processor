--[[
Pattern: CS_US_FUTURE_DATE
DisplayName: CS-US Future Date Note
Description: Serial forms a valid US date (mmddyyyy) that is in the future.
BookRef: CS-560
Tier: 7
Examples: ["12252040", "01012030", "07042050"]
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    local mm = tonumber(d:sub(1, 2))
    local dd = tonumber(d:sub(3, 4))
    local yyyy = tonumber(d:sub(5, 8))

    if not is_valid_date(mm, dd, yyyy) then return {matched = false} end

    local cur_year = ctx.metadata.current_year or 2026
    local cur_month = ctx.metadata.current_month or 1
    local cur_day = ctx.metadata.current_day or 1

    -- Must be strictly in the future
    local is_future = false
    if yyyy > cur_year then
        is_future = true
    elseif yyyy == cur_year then
        if mm > cur_month then
            is_future = true
        elseif mm == cur_month and dd > cur_day then
            is_future = true
        end
    end

    if not is_future then return {matched = false} end

    return {
        matched = true,
        group_boxes = {
            {from = 0, to = 1, color = "coral", thickness = 2},
            {from = 2, to = 3, color = "orange", thickness = 2},
            {from = 4, to = 7, color = "cyan", thickness = 2}
        },
        connectors = {
            {from = 1, to = 2, color = "gold", style = "line"},
            {from = 3, to = 4, color = "gold", style = "line"}
        },
        message = string.format("US Future Date: %02d/%02d/%04d", mm, dd, yyyy)
    }
end
