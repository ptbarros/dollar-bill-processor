--[[
Pattern: CS_STAND_ALONE_DATE
DisplayName: CS-Stand Alone Date
Description: Catch-all: a valid mmdd or ddmm block (4 digits) surrounded by zeros at any position in the serial.
BookRef: CS-1780
Tier: 7
Examples: ["01250000", "00012500", "00001225"]
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
                        {from = s0, to = s0 + 3, color = "lime", thickness = 3}
                    },
                    message = string.format("Stand Alone Date: %02d/%02d at position %d", a, b, s0)
                }
            end

            -- Check as ddmm (EU)
            if is_valid_mmdd(b, a) then
                return {
                    matched = true,
                    group_boxes = {
                        {from = s0, to = s0 + 3, color = "lime", thickness = 3}
                    },
                    message = string.format("Stand Alone Date: %02d/%02d (EU) at position %d", a, b, s0)
                }
            end
        end
    end

    return {matched = false}
end
