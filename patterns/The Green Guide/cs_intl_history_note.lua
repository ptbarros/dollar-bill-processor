--[[
Pattern: CS_INTL_HISTORY_NOTE
DisplayName: CS-INTL History Note
Description: Serial forms a valid INTL date (yyyymmdd) with year more than 100 years ago.
BookRef: CS-640
Tier: 7
Examples: ["17760704", "19001225", "18000101"]
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    local yyyy = tonumber(d:sub(1, 4))
    local mm = tonumber(d:sub(5, 6))
    local dd = tonumber(d:sub(7, 8))

    if not is_valid_date(mm, dd, yyyy) then return {matched = false} end

    local cur_year = ctx.metadata.current_year or 2026
    if yyyy >= cur_year - 100 then return {matched = false} end

    -- Exclude leap year history (handled by CS-650)
    if mm == 2 and dd == 29 then return {matched = false} end

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
        message = string.format("INTL History Note: %04d-%02d-%02d", yyyy, mm, dd)
    }
end
