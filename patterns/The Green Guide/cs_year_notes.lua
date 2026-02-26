--[[
Pattern: CS_YEAR_NOTES
DisplayName: CS-Year Notes
Description: Catch-all: any serial containing a valid 4-digit year (1700-2099) as a consecutive block at any of the 5 possible positions.
BookRef: CS-700
Tier: 8
Examples: ["19751234", "12197534", "12341975"]
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    for start = 1, 5 do
        local year_str = d:sub(start, start + 3)
        local year = tonumber(year_str)

        if year and is_valid_year(year) then
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
