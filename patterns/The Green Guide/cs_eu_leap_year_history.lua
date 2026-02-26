--[[
Pattern: CS_EU_LEAP_YEAR_HISTORY
DisplayName: CS-EU Leap Year History Note
Description: Serial forms EU date (ddmmyyyy) of 29 Feb in a leap year, year more than 100 years ago.
BookRef: CS-600
Tier: 6
Examples: ["29021904", "29021808"]
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    local dd = tonumber(d:sub(1, 2))
    local mm = tonumber(d:sub(3, 4))
    local yyyy = tonumber(d:sub(5, 8))

    if dd ~= 29 or mm ~= 2 then return {matched = false} end
    if not is_valid_date(mm, dd, yyyy) then return {matched = false} end

    local cur_year = ctx.metadata.current_year or 2026
    if yyyy >= cur_year - 100 then return {matched = false} end

    return {
        matched = true,
        group_boxes = {
            {from = 0, to = 1, color = "orange", thickness = 2},
            {from = 2, to = 3, color = "coral", thickness = 2},
            {from = 4, to = 7, color = "cyan", thickness = 2}
        },
        connectors = {
            {from = 1, to = 2, color = "gold", style = "line"},
            {from = 3, to = 4, color = "gold", style = "line"}
        },
        message = string.format("EU Leap Year History: %02d/%02d/%04d", dd, mm, yyyy)
    }
end
