--[[
Pattern: BRIDGED_BOOKEND
Description: Bookend with bridge pattern
Tier: 4
Examples: ["12233221", "34455443"]
Odds: 1 in 12,176
Price: $5-$25
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Pattern: ABCCBA pattern embedded, like AB CC BA or similar
    -- Check for AABBCCDD where first half mirrors second half

    -- Check ABBAABBA pattern
    if digits:sub(1, 1) == digits:sub(4, 4) and digits:sub(1, 1) == digits:sub(5, 5) and digits:sub(1, 1) == digits:sub(8, 8) and
       digits:sub(2, 2) == digits:sub(3, 3) and digits:sub(2, 2) == digits:sub(6, 6) and digits:sub(2, 2) == digits:sub(7, 7) then
        local a = digits:sub(1, 1)
        local b = digits:sub(2, 2)
        return {
            matched = true,
            highlights = {
                highlight({0, 3, 4, 7}, "orange", "A"),
                highlight({1, 2, 5, 6}, "coral", "B")
            },
            connectors = {
                connector(0, 7, "orange", "arc"),
                connector(3, 4, "orange", "line")
            },
            message = "Bridged bookend: " .. a .. b .. b .. a .. a .. b .. b .. a
        }
    end

    -- Check ABCCBA?? pattern (first 6 is palindrome)
    if digits:sub(1, 1) == digits:sub(6, 6) and
       digits:sub(2, 2) == digits:sub(5, 5) and
       digits:sub(3, 3) == digits:sub(4, 4) then
        return {
            matched = true,
            highlights = {
                highlight({0, 5}, "orange", "A"),
                highlight({1, 4}, "coral", "B"),
                highlight({2, 3}, "gold", "CC")
            },
            connectors = {
                connector(0, 5, "orange", "arc"),
                connector(1, 4, "coral", "arc")
            },
            group_boxes = {
                {from = 2, to = 3, color = "gold", thickness = 2}
            },
            message = "Bridged bookend: palindrome bridge"
        }
    end

    -- Check ??ABCCBA pattern (last 6 is palindrome)
    if digits:sub(3, 3) == digits:sub(8, 8) and
       digits:sub(4, 4) == digits:sub(7, 7) and
       digits:sub(5, 5) == digits:sub(6, 6) then
        return {
            matched = true,
            highlights = {
                highlight({2, 7}, "orange", "A"),
                highlight({3, 6}, "coral", "B"),
                highlight({4, 5}, "gold", "CC")
            },
            connectors = {
                connector(2, 7, "orange", "arc"),
                connector(3, 6, "coral", "arc")
            },
            group_boxes = {
                {from = 4, to = 5, color = "gold", thickness = 2}
            },
            message = "Bridged bookend: palindrome bridge"
        }
    end

    return {matched = false}
end
