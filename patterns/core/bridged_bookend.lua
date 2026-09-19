--[[
Pattern: BRIDGED_BOOKEND
DisplayName: Bridged Bookend
Description: First and last digit match (bookend) wrapping a middle bridge that is either 2 triples (AAABBB) or 3 pairs (AABBCC). Uses all 8 digits.
Tier: 4
Examples: ["12223331", "15566771", "19998881", "34455663"]
Price: $5-$25
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Bookend: first and last digit match.
    if d:sub(1, 1) ~= d:sub(8, 8) then return {matched = false} end

    local mid = d:sub(2, 7)  -- the 6 bridge digits

    -- Bridge option A: two triples, AAABBB (A != B).
    local a, b = mid:sub(1, 1), mid:sub(4, 4)
    local two_triples = mid:sub(1, 3) == a:rep(3) and mid:sub(4, 6) == b:rep(3) and a ~= b

    -- Bridge option B: three pairs, AABBCC with adjacent pairs differing.
    local p1, p2, p3 = mid:sub(1, 1), mid:sub(3, 3), mid:sub(5, 5)
    local three_pairs = mid:sub(1, 2) == p1:rep(2) and mid:sub(3, 4) == p2:rep(2)
        and mid:sub(5, 6) == p3:rep(2) and p1 ~= p2 and p2 ~= p3

    if not two_triples and not three_pairs then return {matched = false} end

    local group_boxes = {}
    if two_triples then
        group_boxes = {
            {from = 1, to = 3, color = "orange", thickness = 3},
            {from = 4, to = 6, color = "magenta", thickness = 3}
        }
    else
        group_boxes = {
            {from = 1, to = 2, color = "orange", thickness = 3},
            {from = 3, to = 4, color = "magenta", thickness = 3},
            {from = 5, to = 6, color = "red", thickness = 3}
        }
    end

    local kind = two_triples and "2 triples" or "3 pairs"
    return {
        matched = true,
        message = "Bridged bookend (" .. d:sub(1, 1) .. " wraps " .. kind .. ")",
        highlights = {
            {positions = {0, 7}, color = "blue"}
        },
        connectors = {
            {from = 0, to = 7, color = "blue", style = "arc"}
        },
        group_boxes = group_boxes
    }
end
