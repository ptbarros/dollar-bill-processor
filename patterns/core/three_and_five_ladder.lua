--[[
Pattern: THREE_AND_FIVE_LADDER
DisplayName: Three And Five Ladder
Description: A 3-digit ladder and a 5-digit ladder side by side, filling all 8 digits (e.g. 123 76543).
Tier: 4
Examples: ["12376543", "32187654", "54332101"]
Odds: 1 in 200,000
Price: $10-$100+
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    local function lad(sub) return is_ascending(sub) or is_descending(sub) end

    -- A full 8-digit ladder is its own pattern (Ladder 8), not a 3+5.
    if lad(d) then return {matched = false} end

    -- Two adjacent ladders of lengths 3 and 5 (either order) covering all 8 digits.
    for _, split in ipairs({3, 5}) do
        local a = d:sub(1, split)
        local b = d:sub(split + 1, 8)
        if lad(a) and lad(b) then
            return {
                matched = true,
                message = "3 and 5 ladder",
                highlights = {},
                group_boxes = {
                    {from = 0, to = split - 1, color = "blue", thickness = 3},
                    {from = split, to = 7, color = "orange", thickness = 3}
                }
            }
        end
    end

    return {matched = false}
end
