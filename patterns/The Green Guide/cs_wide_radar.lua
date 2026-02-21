--[[
Pattern: CS_WIDE_RADAR
DisplayName: CS-Wide Radar
Description: CS-60AK structure split as AAAXBBBAAA — first 3 and last 3 are the same digit (A), with a pair or different digit (B) in the center. e.g., 33322333.
BookRef: CS-1290
Tier: 2
Examples: ["33322333", "77700777", "11188111"]
Price: $1,500-$10,000+
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Structure: AAABBAAA (positions 0-2 = A, 3-4 = B pair, 5-7 = A)
    local a1 = d:sub(1, 1)
    local b1 = d:sub(4, 4)

    -- Check first 3 are the same digit A
    if d:sub(2, 2) ~= a1 or d:sub(3, 3) ~= a1 then
        return {matched = false}
    end

    -- Check last 3 are the same digit A
    if d:sub(6, 6) ~= a1 or d:sub(7, 7) ~= a1 or d:sub(8, 8) ~= a1 then
        return {matched = false}
    end

    -- Center 2 digits (positions 3-4) must form a pair of digit B ≠ A
    if d:sub(5, 5) ~= b1 then
        return {matched = false}
    end
    if b1 == a1 then
        return {matched = false}
    end

    -- Verify it's a CS-60AK: 6 occurrences of A
    local counts = count_digits(d)
    if (counts[a1] or 0) ~= 6 then
        return {matched = false}
    end

    return {
        matched = true,
        group_boxes = {
            {from = 0, to = 2, color = "gold", thickness = 3},
            {from = 3, to = 4, color = "coral", thickness = 2},
            {from = 5, to = 7, color = "gold", thickness = 3},
        },
        connectors = {
            {from = 0, to = 7, color = "gold", style = "arc"},
        },
        message = a1 .. a1 .. a1 .. " + " .. b1 .. b1 .. " + " .. a1 .. a1 .. a1 .. " (CS-Wide Radar CS-1290)"
    }
end
