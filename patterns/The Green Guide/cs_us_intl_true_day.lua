--[[
Pattern: CS_US_INTL_TRUE_DAY
DisplayName: CS-US & INTL True Day Notes
Description: A valid mmdd block (US/INTL format) at any position, with the remaining 4 digits all zeros.
BookRef: CS-770
Tier: 7
Examples: ["12250000", "00001225", "00122500"]
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    for start = 1, 5 do
        local block = d:sub(start, start + 3)
        local mm = tonumber(block:sub(1, 2))
        local dd = tonumber(block:sub(3, 4))

        if is_valid_mmdd(mm, dd) then
            local rest = d:sub(1, start - 1) .. d:sub(start + 4)
            if rest == string.rep("0", #rest) then
                local s0 = start - 1
                return {
                    matched = true,
                    group_boxes = {
                        {from = s0, to = s0 + 1, color = "orange", thickness = 3},
                        {from = s0 + 2, to = s0 + 3, color = "coral", thickness = 3}
                    },
                    message = string.format("US/INTL True Day: %02d/%02d with all zeros", mm, dd)
                }
            end
        end
    end

    return {matched = false}
end
