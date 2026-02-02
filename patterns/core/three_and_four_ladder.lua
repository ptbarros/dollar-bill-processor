--[[
Pattern: THREE_AND_FOUR_LADDER
Description: 3-digit and 4-digit ladder combined
Tier: 4
Examples: ["12376543", "32187654"]
Odds: 1 in 22,222
Price: $10-$100+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Find all ladders of length 3+ and 4+
    local ladder3 = find_ladder_of_length(digits, 3)
    local ladder4 = find_ladder_of_length(digits, 4)

    if not ladder3 or not ladder4 then
        return {matched = false}
    end

    -- They should be non-overlapping
    local l3_end = ladder3.start + ladder3.length - 1
    local l4_end = ladder4.start + ladder4.length - 1

    -- Check for overlap
    local overlapping = not (l3_end < ladder4.start or l4_end < ladder3.start)

    if overlapping then
        -- Try to find another ladder that doesn't overlap
        return {matched = false}
    end

    local pos3 = {}
    for i = 0, ladder3.length - 1 do
        table.insert(pos3, ladder3.start + i)
    end

    local pos4 = {}
    for i = 0, ladder4.length - 1 do
        table.insert(pos4, ladder4.start + i)
    end

    return {
        matched = true,
        highlights = {
            highlight(pos3, "lime", "3-ladder"),
            highlight(pos4, "teal", "4-ladder")
        },
        connectors = {},
        message = "3-ladder + 4-ladder"
    }
end
