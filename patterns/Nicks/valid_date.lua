--[[
Pattern: VALID_DATE
DisplayName: Valid Date
Description: Reads as a calendar date — either month/day/year or year/month/day — with the year between 1500 and 2050 (e.g. 12251985, 19850704).
Tier: 7
Odds: 1 in 239 (402,498 per 96M)
Examples: ["01011990", "12251985", "19850704", "20001231"]
--]]

local MONTH_NAMES = {"January", "February", "March", "April", "May", "June",
                     "July", "August", "September", "October", "November", "December"}

local function days_in_month(m, y)
    if m == 2 then
        local leap = (y % 4 == 0 and (y % 100 ~= 0 or y % 400 == 0))
        return leap and 29 or 28
    elseif m == 4 or m == 6 or m == 9 or m == 11 then
        return 30
    else
        return 31
    end
end

-- Per-month day limit (Ed review), not a flat 31.
local function valid_ymd(y, m, d)
    if not (y and m and d) then return false end
    if y < 1500 or y > 2050 then return false end
    if m < 1 or m > 12 then return false end
    if d < 1 or d > days_in_month(m, y) then return false end
    return true
end

function match(ctx)
    local s = ctx.digits

    -- MMDDYYYY
    local month = tonumber(s:sub(1, 2))
    local day = tonumber(s:sub(3, 4))
    local year = tonumber(s:sub(5, 8))
    if valid_ymd(year, month, day) then
        return {
            matched = true,
            message = string.format("%s %d, %d (MMDDYYYY)", MONTH_NAMES[month], day, year),
            highlights = {
                {positions = {0, 1}, color = "cyan"},
                {positions = {2, 3}, color = "lime"},
                {positions = {4, 5, 6, 7}, color = "gold"}
            }
        }
    end

    -- YYYYMMDD
    year = tonumber(s:sub(1, 4))
    month = tonumber(s:sub(5, 6))
    day = tonumber(s:sub(7, 8))
    if valid_ymd(year, month, day) then
        return {
            matched = true,
            message = string.format("%s %d, %d (YYYYMMDD)", MONTH_NAMES[month], day, year),
            highlights = {
                {positions = {0, 1, 2, 3}, color = "gold"},
                {positions = {4, 5}, color = "cyan"},
                {positions = {6, 7}, color = "lime"}
            }
        }
    end

    return {matched = false}
end
