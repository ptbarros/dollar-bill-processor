--[[
Pattern: CONSECUTIVE_PAIRS_LADDER
DisplayName: 3 Consecutive Pairs Ladder
Description: Three identical pairs in a row (AABBCC) anywhere in the serial, where A, B, C count by 1 (e.g. 88 77 66).
Tier: 5
Odds: 1 in 21,039 (4,563 per 96M)
Examples: ["11223344", "98877663", "22334455", "55443322"]
--]]

function match(ctx)
    local s = ctx.digits
    if #s ~= 8 then return {matched = false} end

    -- Slide a 6-digit window (start positions 0,1,2) looking for AABBCC where the
    -- three pair-digits count up or down by 1.
    for start = 0, 2 do
        local a1, a2 = s:sub(start + 1, start + 1), s:sub(start + 2, start + 2)
        local b1, b2 = s:sub(start + 3, start + 3), s:sub(start + 4, start + 4)
        local c1, c2 = s:sub(start + 5, start + 5), s:sub(start + 6, start + 6)

        if a1 == a2 and b1 == b2 and c1 == c2 then
            local a, b, c = tonumber(a1), tonumber(b1), tonumber(c1)
            local is_asc = (b == a + 1 and c == b + 1)
            local is_desc = (b == a - 1 and c == b - 1)
            if is_asc or is_desc then
                local direction = is_asc and "ascending" or "descending"
                return {
                    matched = true,
                    message = "3 consecutive pairs " .. direction .. " ladder: "
                        .. a1 .. a1 .. " " .. b1 .. b1 .. " " .. c1 .. c1,
                    highlights = {},
                    group_boxes = {
                        {from = start,     to = start + 1, color = "blue", thickness = 3},
                        {from = start + 2, to = start + 3, color = "orange", thickness = 3},
                        {from = start + 4, to = start + 5, color = "magenta", thickness = 3}
                    }
                }
            end
        end
    end

    return {matched = false}
end
