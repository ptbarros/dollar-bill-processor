--[[
Pattern: HYBRID_NOTES
Description: The first two digits repeat at the end in either order, with a repeated pair filling the middle (e.g. 32·6868·32).
Tier: 6
Examples: ["32686832", "32686823", "12525212", "12525221"]
Odds: 1 in 6,130 (15,660 per 96M)
Price: $10-$50+
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then
        return {matched = false}
    end

    -- Exactly the four shapes ABDCDCAB / ABDCDCBA / ABACACAB / ABACACBA
    -- (Ed review): outer pair AB repeats at the end in either order, and the middle
    -- four are a pair repeated (positions 3=5, 4=6). Dropped the old palindrome
    -- and ABABABCD branches.
    local a, b = d:sub(1, 1), d:sub(2, 2)
    if a == b then
        return {matched = false}
    end

    -- Middle four = a repeated pair: positions 3&5 match, 4&6 match, and the pair
    -- is two distinct digits.
    local m1, m2 = d:sub(3, 3), d:sub(4, 4)
    if not (d:sub(5, 5) == m1 and d:sub(6, 6) == m2 and m1 ~= m2) then
        return {matched = false}
    end

    -- Outer pair repeats at the end, same order (..AB) or reversed (..BA).
    local e1, e2 = d:sub(7, 7), d:sub(8, 8)
    if not ((e1 == a and e2 == b) or (e1 == b and e2 == a)) then
        return {matched = false}
    end

    return {
        matched = true,
        highlights = {
            {positions = {0, 1, 6, 7}, color = "blue", label = "outer pair"},
            {positions = {2, 3, 4, 5}, color = "orange", label = "repeated middle"}
        },
        group_boxes = {
            {from = 0, to = 1, color = "blue", thickness = 3},
            {from = 6, to = 7, color = "blue", thickness = 3},
            {from = 2, to = 5, color = "orange", thickness = 3}
        },
        connectors = {},
        message = "Hybrid: outer pair " .. a .. b .. " + repeated middle " .. m1 .. m2
    }
end
