--[[
Pattern: CS_STAND_ALONE_YEAR
DisplayName: CS-Stand Alone Year
Description: A 4-digit year (1000-2099) as a consecutive block, with remaining 4 digits all zeros. e.g., M 01975000 M.
BookRef: CS-1810
Tier: 5
Examples: ["01975000", "00197500", "19760000"]
Odds: 1 in 10,000
Price: $5-$20
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Scan all possible starting positions for a 4-digit year block (Lua 1-indexed, start = 1..5)
    for start = 1, 5 do
        local year_str = d:sub(start, start + 3)
        local year = tonumber(year_str)

        -- Year must be in range 1000-2099
        if year and year >= 1000 and year <= 2099 then
            -- Remaining digits must all be zeros
            local rest = d:sub(1, start - 1) .. d:sub(start + 4)
            if rest == string.rep("0", #rest) then
                -- 0-indexed positions for the year block
                local year_start = start - 1
                local year_end = start + 2  -- 0-indexed inclusive

                return {
                    matched = true,
                    group_boxes = {
                        {from = year_start, to = year_end, color = "gold", thickness = 3}
                    },
                    message = "Stand alone year " .. year .. " with all zeros (CS-Stand Alone Year)"
                }
            end
        end
    end

    return {matched = false}
end
