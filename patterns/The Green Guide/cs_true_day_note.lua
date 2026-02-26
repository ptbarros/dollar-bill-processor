--[[
Pattern: CS_TRUE_DAY_NOTE
DisplayName: CS-True Day Note
Description: A valid mmdd or ddmm block at any position, with the remaining 4 digits all zeros.
BookRef: CS-760
Tier: 7
Examples: ["12250000", "00001225", "00122500"]
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    for start = 1, 5 do
        local block = d:sub(start, start + 3)
        local a = tonumber(block:sub(1, 2))
        local b = tonumber(block:sub(3, 4))

        local rest = d:sub(1, start - 1) .. d:sub(start + 4)
        if rest == string.rep("0", #rest) then
            local s0 = start - 1

            -- Check as mmdd (US/INTL)
            if is_valid_mmdd(a, b) then
                return {
                    matched = true,
                    group_boxes = {
                        {from = s0, to = s0 + 1, color = "orange", thickness = 3},
                        {from = s0 + 2, to = s0 + 3, color = "coral", thickness = 3}
                    },
                    message = string.format("True Day Note: %02d/%02d with all zeros", a, b)
                }
            end

            -- Check as ddmm (EU)
            if is_valid_mmdd(b, a) then
                return {
                    matched = true,
                    group_boxes = {
                        {from = s0, to = s0 + 1, color = "coral", thickness = 3},
                        {from = s0 + 2, to = s0 + 3, color = "orange", thickness = 3}
                    },
                    message = string.format("True Day Note: %02d/%02d (EU) with all zeros", a, b)
                }
            end
        end
    end

    return {matched = false}
end
