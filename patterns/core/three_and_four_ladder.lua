--[[
Pattern: THREE_AND_FOUR_LADDER
Description: 3-digit and 4-digit ladder combined
Tier: 4
Examples: ["12387650", "32145670"]
Odds: 1 in 22,222
Price: $10-$100+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- A clean 3+5 ladder split is the dedicated Three And Five Ladder; defer to it
    -- (Ed review: this one was also catching 3-and-5).
    local function lad(sub) return is_ascending(sub) or is_descending(sub) end
    if (lad(digits:sub(1, 3)) and lad(digits:sub(4, 8))) or
       (lad(digits:sub(1, 5)) and lad(digits:sub(6, 8))) then
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
        -- One box around each ladder, no per-digit boxes (Ed review).
        highlights = {},
        group_boxes = {
            {from = pos3[1], to = pos3[#pos3], color = "blue", thickness = 3},
            {from = pos4[1], to = pos4[#pos4], color = "orange", thickness = 3}
        },
        connectors = {},
        message = "3-ladder + 4-ladder"
    }
end
