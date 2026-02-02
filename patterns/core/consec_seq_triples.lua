--[[
Pattern: CONSEC_SEQ_TRIPLES
Description: Consecutive sequential triples (AAABBBXX)
Tier: 4
Examples: ["11122234", "22233345"]
Odds: 1 in 24,691
Price: $5-$15
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check for AAA BBB pattern where A and B are consecutive
    -- First 3 same
    local a = digits:sub(1, 1)
    if digits:sub(2, 2) ~= a or digits:sub(3, 3) ~= a then
        return {matched = false}
    end

    -- Next 3 same
    local b = digits:sub(4, 4)
    if digits:sub(5, 5) ~= b or digits:sub(6, 6) ~= b then
        return {matched = false}
    end

    -- A and B must be consecutive
    local a_num = tonumber(a)
    local b_num = tonumber(b)
    if b_num ~= a_num + 1 and b_num ~= a_num - 1 then
        return {matched = false}
    end

    local direction = b_num > a_num and "ascending" or "descending"

    return {
        matched = true,
        highlights = {
            highlight({0, 1, 2}, "gold", "first triple"),
            highlight({3, 4, 5}, "coral", "second triple")
        },
        group_boxes = {
            {from = 0, to = 2, color = "gold", thickness = 2},
            {from = 3, to = 5, color = "coral", thickness = 2}
        },
        connectors = {},
        message = "Sequential triples " .. direction .. ": " .. a .. a .. a .. b .. b .. b
    }
end
