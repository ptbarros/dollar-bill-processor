--[[
Pattern: VALID_DATE
DisplayName: Valid Date
Description: Middle 8 digits form a valid date (MMDDYYYY or YYYYMMDD)
Tier: 7
Examples: ["01011990", "12251985", "19850704", "20001231"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Try MMDDYYYY format
    local month = tonumber(s:sub(1, 2))
    local day = tonumber(s:sub(3, 4))
    local year = tonumber(s:sub(5, 8))

    if month and day and year then
        if month >= 1 and month <= 12 and day >= 1 and day <= 31 and year >= 1500 and year <= 2050 then
            local month_names = {"January", "February", "March", "April", "May", "June",
                                 "July", "August", "September", "October", "November", "December"}
            return {
                matched = true,
                message = string.format("%s %d, %d (MMDDYYYY)", month_names[month], day, year),
                highlights = {
                    {positions = {0, 1}, color = "cyan"},
                    {positions = {2, 3}, color = "lime"},
                    {positions = {4, 5, 6, 7}, color = "gold"}
                }
            }
        end
    end

    -- Try YYYYMMDD format
    year = tonumber(s:sub(1, 4))
    month = tonumber(s:sub(5, 6))
    day = tonumber(s:sub(7, 8))

    if month and day and year then
        if year >= 1500 and year <= 2050 and month >= 1 and month <= 12 and day >= 1 and day <= 31 then
            local month_names = {"January", "February", "March", "April", "May", "June",
                                 "July", "August", "September", "October", "November", "December"}
            return {
                matched = true,
                message = string.format("%s %d, %d (YYYYMMDD)", month_names[month], day, year),
                highlights = {
                    {positions = {0, 1, 2, 3}, color = "gold"},
                    {positions = {4, 5}, color = "cyan"},
                    {positions = {6, 7}, color = "lime"}
                }
            }
        end
    end

    return {matched = false}
end
