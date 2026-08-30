--[[
Pattern: CS_YEAR_NOTES
DisplayName: CS-Year Notes
Description: Catch-all: any serial containing a valid 4-digit year (default 1700-2099, editable below) as a consecutive block at any of the 5 possible positions.
BookRef: CS-700
Tier: 8
Examples: ["19751234", "12197534", "12341975"]
--]]

function match(ctx)
    -- === Editable year range (inclusive) ===
    local YEAR_MIN = 1700   -- earliest year to accept
    local YEAR_MAX = 2099   -- latest year to accept
    -- =======================================

    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    for start = 1, 5 do
        local year_str = d:sub(start, start + 3)
        local year = tonumber(year_str)

        if year and (year >= YEAR_MIN and year <= YEAR_MAX) then
            local s0 = start - 1
            return {
                matched = true,
                group_boxes = {
                    {from = s0, to = s0 + 3, color = "cyan", thickness = 3}
                },
                message = "Year Note: " .. year_str .. " at position " .. s0
            }
        end
    end

    return {matched = false}
end
