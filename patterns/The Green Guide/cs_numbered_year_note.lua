--[[
Pattern: CS_NUMBERED_YEAR_NOTE
DisplayName: CS-Numbered Year Note
Description: A valid 4-digit year (default 1700-2099, editable below) at any position, with the remaining 4 digits all the same non-zero digit (4OAK).
BookRef: CS-680
Tier: 7
Examples: ["19753333", "33319753", "33197533"]
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
            if #rest == 4 then
                local first = rest:sub(1, 1)
                if first ~= "0" and rest == string.rep(first, 4) then
                    local s0 = start - 1
                    -- Highlight the non-year digits
                    local other_pos = {}
                    for i = 0, 7 do
                        if i < s0 or i > s0 + 3 then
                            table.insert(other_pos, i)
                        end
                    end
                    return {
                        matched = true,
                        group_boxes = {
                            {from = s0, to = s0 + 3, color = "cyan", thickness = 3}
                        },
                        highlights = {
                            {positions = other_pos, color = "lime"}
                        },
                        message = "Numbered Year Note: " .. year_str .. " with " .. first .. first .. first .. first
                    }
                end
            end
        end
    end

    return {matched = false}
end
