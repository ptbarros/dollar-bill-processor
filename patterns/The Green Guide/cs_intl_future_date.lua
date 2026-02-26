--[[
Pattern: CS_INTL_FUTURE_DATE
DisplayName: CS-INTL Future Date Note
Description: Serial forms a valid INTL date (yyyymmdd) that is in the future.
BookRef: CS-660
Tier: 7
Examples: ["20401225", "20300101", "20500704"]
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    local yyyy = tonumber(d:sub(1, 4))
    local mm = tonumber(d:sub(5, 6))
    local dd = tonumber(d:sub(7, 8))

    if not is_valid_date(mm, dd, yyyy) then return {matched = false} end

    local cur_year = ctx.metadata.current_year or 2026
    local cur_month = ctx.metadata.current_month or 1
    local cur_day = ctx.metadata.current_day or 1

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
            {from = 0, to = 3, color = "cyan", thickness = 2},
            {from = 4, to = 5, color = "coral", thickness = 2},
            {from = 6, to = 7, color = "orange", thickness = 2}
        },
        connectors = {
            {from = 3, to = 4, color = "gold", style = "line"},
            {from = 5, to = 6, color = "gold", style = "line"}
        },
        message = string.format("INTL Future Date: %04d-%02d-%02d", yyyy, mm, dd)
    }
end
