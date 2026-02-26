--[[
Pattern: CS_INTL_LEAP_YEAR_HISTORY
DisplayName: CS-INTL Leap Year History Note
Description: Serial forms INTL date (yyyymmdd) of Feb 29 in a leap year, year more than 100 years ago.
BookRef: CS-650
Tier: 6
Examples: ["19040229", "18080229"]
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    local yyyy = tonumber(d:sub(1, 4))
    local mm = tonumber(d:sub(5, 6))
    local dd = tonumber(d:sub(7, 8))

    if mm ~= 2 or dd ~= 29 then return {matched = false} end
    if not is_valid_date(mm, dd, yyyy) then return {matched = false} end

    local cur_year = ctx.metadata.current_year or 2026
    if yyyy >= cur_year - 100 then return {matched = false} end

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
        message = string.format("INTL Leap Year History: %04d-%02d-%02d", yyyy, mm, dd)
    }
end
