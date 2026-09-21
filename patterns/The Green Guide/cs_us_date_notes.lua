--[[
Pattern: CS_US_DATE_NOTES
DisplayName: Date Note M/D/Y
Description: Reads as a real month-day-year birthday in the month-first style, from within the last hundred years (e.g. 11·22·1975).
BookRef: CS-520
Tier: 7
Odds: Depends on the date
Examples: ["11221975", "07041950", "01012000"]
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    local mm = tonumber(d:sub(1, 2))
    local dd = tonumber(d:sub(3, 4))
    local yyyy = tonumber(d:sub(5, 8))

    if not is_valid_date(mm, dd, yyyy) then return {matched = false} end

    local cur_year = ctx.metadata.current_year or 2026
    -- Birthday: year in [current_year - 100, current_year]
    if yyyy < cur_year - 100 or yyyy > cur_year then return {matched = false} end

    -- Exclude leap year birthdays (handled by CS-530)
    if mm == 2 and dd == 29 then return {matched = false} end

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
        message = string.format("US Birthday: %02d/%02d/%04d", mm, dd, yyyy)
    }
end
