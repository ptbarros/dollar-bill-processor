--[[
Pattern: CS_US_LEAP_YEAR_BIRTHDAY
DisplayName: CS-US Leap Year Birthday Note
Description: Serial forms US date (mmddyyyy) of Feb 29 in a leap year, year within the last 100 years.
BookRef: CS-530
Tier: 6
Examples: ["02292000", "02291984", "02292024"]
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    local mm = tonumber(d:sub(1, 2))
    local dd = tonumber(d:sub(3, 4))
    local yyyy = tonumber(d:sub(5, 8))

    if mm ~= 2 or dd ~= 29 then return {matched = false} end
    if not is_valid_date(mm, dd, yyyy) then return {matched = false} end

    local cur_year = ctx.metadata.current_year or 2026
    if yyyy < cur_year - 100 or yyyy > cur_year then return {matched = false} end

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
        message = string.format("US Leap Year Birthday: %02d/%02d/%04d", mm, dd, yyyy)
    }
end
