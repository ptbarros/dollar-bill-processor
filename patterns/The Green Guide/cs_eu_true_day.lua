--[[
Pattern: CS_EU_TRUE_DAY
DisplayName: CS-EU True Day Notes
Description: A valid ddmm block (EU format) at any position, with the remaining 4 digits all zeros.
BookRef: CS-780
Tier: 7
Examples: ["25120000", "00002512", "00251200"]
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
                    message = string.format("EU True Day: %02d/%02d (dd/mm) with all zeros", dd, mm)
                }
            end
        end
    end

    return {matched = false}
end
