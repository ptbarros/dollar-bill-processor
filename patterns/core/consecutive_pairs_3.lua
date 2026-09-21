--[[
Pattern: CONSECUTIVE_PAIRS_3
DisplayName: 3 Consecutive Pairs
Description: 3 consecutive pairs within the serial
Tier: 7
Odds: 1 in 433 (221,859 per 96M)
Examples: ["11223312", "12233445", "00112234", "99887700"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Look for 3 consecutive pairs (6 digits)
    for start = 1, 3 do
        local valid = true
        for i = start, start + 4, 2 do
            local pair = s:sub(i, i + 1)
            if pair:sub(1, 1) ~= pair:sub(2, 2) then
                valid = false
                break
            end
            -- Make sure next pair is different
            if i + 2 <= start + 4 then
                local next_pair = s:sub(i + 2, i + 3)
                if pair:sub(1, 1) == next_pair:sub(1, 1) then
                    valid = false
                    break
                end
            end
        end

        if valid then
            -- One group box around each of the 3 consecutive pairs (Ed review);
            -- no per-digit boxes.
            local base = start - 1
            return {
                matched = true,
                message = "3 consecutive pairs",
                highlights = {},
                group_boxes = {
                    {from = base, to = base + 1, color = "blue", thickness = 3},
                    {from = base + 2, to = base + 3, color = "orange", thickness = 3},
                    {from = base + 4, to = base + 5, color = "magenta", thickness = 3}
                }
            }
        end
    end

    return {matched = false}
end
