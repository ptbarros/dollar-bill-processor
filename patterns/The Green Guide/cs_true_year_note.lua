--[[
Pattern: CS_TRUE_YEAR_NOTE
DisplayName: CS-True Year Note
Description: A valid 4-digit year (default 1700-2099, editable below) at any position in the serial, with the remaining 4 digits all zeros.
BookRef: CS-670
Tier: 6
Examples: ["19750000", "00019750", "00197500"]
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
            local rest = d:sub(1, start - 1) .. d:sub(start + 4)
            if rest == string.rep("0", #rest) then
                local s0 = start - 1
                return {
                    matched = true,
                    group_boxes = {
                        {from = s0, to = s0 + 3, color = "cyan", thickness = 3}
                    },
                    message = "True Year Note: " .. year_str .. " with all zeros"
                }
            end
        end
    end

    return {matched = false}
end
