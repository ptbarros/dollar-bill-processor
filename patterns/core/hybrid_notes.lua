--[[
Pattern: HYBRID_NOTES
Description: Complex hybrid patterns (ABDCDCAB, etc.)
Tier: 4
Examples: ["12343421", "12343412"]
Odds: 1 in 8,333
Price: $10-$50+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Look for ABCDCDAB or ABDCDCAB patterns
    -- These are patterns where part of the serial mirrors/repeats in a complex way

    -- Pattern 1: ABCDCDAB - positions 2,3 repeat at 4,5 and 0,1 repeat at 6,7
    if digits:sub(3, 4) == digits:sub(5, 6) and digits:sub(1, 2) == digits:sub(7, 8) then
        return {
            matched = true,
            highlights = {
                highlight({0, 1}, "orange", "AB start"),
                highlight({2, 3}, "coral", "CD"),
                highlight({4, 5}, "coral", "CD repeat"),
                highlight({6, 7}, "orange", "AB end")
            },
            connectors = {
                connector(0, 6, "orange", "arc"),
                connector(1, 7, "orange", "arc"),
                connector(2, 4, "coral", "arc"),
                connector(3, 5, "coral", "arc")
            },
            message = "Hybrid: ABCDCDAB"
        }
    end

    -- Pattern 2: ABCDDCBA (palindrome with doubled middle)
    if digits:sub(4, 4) == digits:sub(5, 5) and
       digits:sub(1, 1) == digits:sub(8, 8) and
       digits:sub(2, 2) == digits:sub(7, 7) and
       digits:sub(3, 3) == digits:sub(6, 6) then
        return {
            matched = true,
            highlights = {
                highlight({0, 7}, "orange", "A"),
                highlight({1, 6}, "coral", "B"),
                highlight({2, 5}, "magenta", "C"),
                highlight({3, 4}, "gold", "DD")
            },
            connectors = {
                connector(0, 7, "orange", "arc"),
                connector(1, 6, "coral", "arc"),
                connector(2, 5, "magenta", "arc")
            },
            group_boxes = {
                {from = 3, to = 4, color = "gold", thickness = 2}
            },
            message = "Hybrid: ABCDDCBA"
        }
    end

    -- Pattern 3: ABABABCD - super repeater start with different end
    if digits:sub(1, 2) == digits:sub(3, 4) and digits:sub(1, 2) == digits:sub(5, 6) then
        return {
            matched = true,
            highlights = {
                highlight({0, 1, 2, 3, 4, 5}, "magenta", "AB repeat"),
                highlight({6, 7}, "gold", "CD")
            },
            connectors = {
                connector(0, 2, "magenta", "line"),
                connector(2, 4, "magenta", "line")
            },
            message = "Hybrid: ABABABCD"
        }
    end

    return {matched = false}
end
