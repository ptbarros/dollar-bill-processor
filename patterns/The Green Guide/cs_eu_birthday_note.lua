--[[
Pattern: CS_EU_BIRTHDAY_NOTE
DisplayName: CS-EU Birthday Note
Description: Serial forms a valid EU date (ddmmyyyy) as a birthday — year within the last 100 years.
BookRef: CS-570
Tier: 7
Examples: ["22111975", "04071950", "01012000"]
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    local dd = tonumber(d:sub(1, 2))
    local mm = tonumber(d:sub(3, 4))
    local yyyy = tonumber(d:sub(5, 8))

    if not is_valid_date(mm, dd, yyyy) then return {matched = false} end

    local cur_year = ctx.metadata.current_year or 2026
    if yyyy < cur_year - 100 or yyyy > cur_year then return {matched = false} end

    -- Exclude leap year birthdays (handled by CS-580)
    if mm == 2 and dd == 29 then return {matched = false} end

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
        message = string.format("EU Birthday: %02d/%02d/%04d", dd, mm, yyyy)
    }
end
