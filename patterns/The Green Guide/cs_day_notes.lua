--[[
Pattern: CS_DAY_NOTES
DisplayName: CS-Day Notes
Description: Catch-all: any serial containing a valid mmdd or ddmm block (4 digits) at any of the 5 possible positions.
BookRef: CS-750
Tier: 8
Examples: ["12251234", "12122512", "12341225"]
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    for start = 1, 5 do
        local block = d:sub(start, start + 3)
        local a = tonumber(block:sub(1, 2))
        local b = tonumber(block:sub(3, 4))

        -- Check as mmdd (US/INTL)
        if is_valid_mmdd(a, b) then
            local s0 = start - 1
            return {
                matched = true,
                group_boxes = {
                    {from = s0, to = s0 + 1, color = "orange", thickness = 2},
                    {from = s0 + 2, to = s0 + 3, color = "coral", thickness = 2}
                },
                message = string.format("Day Note: %02d/%02d (US) at position %d", a, b, s0)
            }
        end

        -- Check as ddmm (EU)
        if is_valid_mmdd(b, a) then
            local s0 = start - 1
            return {
                matched = true,
                group_boxes = {
                    {from = s0, to = s0 + 1, color = "coral", thickness = 2},
                    {from = s0 + 2, to = s0 + 3, color = "orange", thickness = 2}
                },
                message = string.format("Day Note: %02d/%02d (EU) at position %d", a, b, s0)
            }
        end
    end

    return {matched = false}
end
