--[[
Pattern: CS_STAND_ALONE_US_DATE_YEAR
DisplayName: CS-Stand Alone US Date Year
Description: A 6-digit mmddyy block (US format) surrounded by zeros. yy>=30 maps to 1900+yy, yy<30 maps to 2000+yy.
BookRef: CS-1830
Tier: 7
Examples: ["01225710", "00122571", "12257100"]
--]]

local function map_yy(yy)
    if yy >= 30 then return 1900 + yy else return 2000 + yy end
end

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    for start = 1, 3 do
        local block = d:sub(start, start + 5)
        local rest = d:sub(1, start - 1) .. d:sub(start + 6)
        if rest == string.rep("0", #rest) then
            local mm = tonumber(block:sub(1, 2))
            local dd = tonumber(block:sub(3, 4))
            local yy = tonumber(block:sub(5, 6))
            local yyyy = map_yy(yy)

            if is_valid_date(mm, dd, yyyy) then
                local s0 = start - 1
                return {
                    matched = true,
                    group_boxes = {
                        {from = s0, to = s0 + 1, color = "coral", thickness = 3},
                        {from = s0 + 2, to = s0 + 3, color = "orange", thickness = 3},
                        {from = s0 + 4, to = s0 + 5, color = "cyan", thickness = 3}
                    },
                    message = string.format("Stand Alone US Date Year: %02d/%02d/%02d (%04d)", mm, dd, yy, yyyy)
                }
            end
        end
    end

    return {matched = false}
end
