--[[
Pattern: CS_STAND_ALONE_DATE_EU
DisplayName: CS-Stand Alone Date EU
Description: A valid ddmm block (EU format) surrounded by zeros.
BookRef: CS-1800
Tier: 7
Examples: ["02512000", "00251200", "00002512"]
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    for start = 1, 5 do
        local block = d:sub(start, start + 3)
        local dd = tonumber(block:sub(1, 2))
        local mm = tonumber(block:sub(3, 4))

        if is_valid_mmdd(mm, dd) then
            local rest = d:sub(1, start - 1) .. d:sub(start + 4)
            if rest == string.rep("0", #rest) then
                local s0 = start - 1
                return {
                    matched = true,
                    group_boxes = {
                        {from = s0, to = s0 + 1, color = "coral", thickness = 3},
                        {from = s0 + 2, to = s0 + 3, color = "orange", thickness = 3}
                    },
                    message = string.format("Stand Alone EU Date: %02d/%02d (dd/mm)", dd, mm)
                }
            end
        end
    end

    return {matched = false}
end
