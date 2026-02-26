--[[
Pattern: CS_STAND_ALONE_DATE_YEAR
DisplayName: CS-Stand Alone Date Year
Description: Catch-all: a 6-digit date block (mmddyy, ddmmyy, or yymmdd) surrounded by zeros. 2-digit year: yy>=30 maps to 1900+yy, yy<30 maps to 2000+yy.
BookRef: CS-1820
Tier: 7
Examples: ["01225710", "00122571", "71122500"]
--]]

local function map_yy(yy)
    if yy >= 30 then return 1900 + yy else return 2000 + yy end
end

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- 6-digit block can start at positions 1, 2, or 3 (leaving 2 zeros)
    for start = 1, 3 do
        local block = d:sub(start, start + 5)
        local rest = d:sub(1, start - 1) .. d:sub(start + 6)
        if rest == string.rep("0", #rest) then
            local s0 = start - 1

            -- Try mmddyy (US)
            local mm = tonumber(block:sub(1, 2))
            local dd = tonumber(block:sub(3, 4))
            local yy = tonumber(block:sub(5, 6))
            local yyyy = map_yy(yy)
            if is_valid_date(mm, dd, yyyy) then
                return {
                    matched = true,
                    group_boxes = {
                        {from = s0, to = s0 + 5, color = "lime", thickness = 3}
                    },
                    message = string.format("Stand Alone Date Year: %02d/%02d/%02d (%04d, US)", mm, dd, yy, yyyy)
                }
            end

            -- Try ddmmyy (EU)
            dd = tonumber(block:sub(1, 2))
            mm = tonumber(block:sub(3, 4))
            yy = tonumber(block:sub(5, 6))
            yyyy = map_yy(yy)
            if is_valid_date(mm, dd, yyyy) then
                return {
                    matched = true,
                    group_boxes = {
                        {from = s0, to = s0 + 5, color = "lime", thickness = 3}
                    },
                    message = string.format("Stand Alone Date Year: %02d/%02d/%02d (%04d, EU)", dd, mm, yy, yyyy)
                }
            end

            -- Try yymmdd (INTL)
            yy = tonumber(block:sub(1, 2))
            mm = tonumber(block:sub(3, 4))
            dd = tonumber(block:sub(5, 6))
            yyyy = map_yy(yy)
            if is_valid_date(mm, dd, yyyy) then
                return {
                    matched = true,
                    group_boxes = {
                        {from = s0, to = s0 + 5, color = "lime", thickness = 3}
                    },
                    message = string.format("Stand Alone Date Year: %02d/%02d/%02d (%04d, INTL)", yy, mm, dd, yyyy)
                }
            end
        end
    end

    return {matched = false}
end
