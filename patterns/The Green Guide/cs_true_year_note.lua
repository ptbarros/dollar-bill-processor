--[[
Pattern: CS_TRUE_YEAR_NOTE
DisplayName: CS-True Year Note
Description: A valid 4-digit year (1700-2099) at any position in the serial, with the remaining 4 digits all zeros.
BookRef: CS-670
Tier: 6
Examples: ["19750000", "00019750", "00197500"]
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    for start = 1, 5 do
        local year_str = d:sub(start, start + 3)
        local year = tonumber(year_str)

        if year and is_valid_year(year) then
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
