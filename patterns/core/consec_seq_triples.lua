--[[
Pattern: CONSEC_SEQ_TRIPLES
Description: Two consecutive sequential triples, at the front (AAABBBXX) or the back (XXAAABBB), the two digits one apart (e.g. 111·222·xx, xx·888·999).
Tier: 4
Examples: ["11122234", "22233345", "34111222", "98999888"]
Odds: 1 in 28,005 (3,428 per 96M)
Price: $5-$15
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Two consecutive triples (AAABBB), front-loaded (base 0 = AAABBBXX) or at the
    -- back (base 2 = XXAAABBB), with the two digits one apart (Ed review).
    for _, base in ipairs({0, 2}) do
        local a = digits:sub(base + 1, base + 1)
        local b = digits:sub(base + 4, base + 4)
        local same_a = a == digits:sub(base + 2, base + 2) and a == digits:sub(base + 3, base + 3)
        local same_b = b == digits:sub(base + 5, base + 5) and b == digits:sub(base + 6, base + 6)
        if same_a and same_b then
            local an, bn = tonumber(a), tonumber(b)
            if bn == an + 1 or bn == an - 1 then
                local direction = bn > an and "ascending" or "descending"
                return {
                    matched = true,
                    highlights = {},
                    group_boxes = {
                        {from = base,     to = base + 2, color = "gold", thickness = 2},
                        {from = base + 3, to = base + 5, color = "coral", thickness = 2}
                    },
                    connectors = {},
                    message = "Sequential triples " .. direction .. ": " .. a .. a .. a .. b .. b .. b
                }
            end
        end
    end

    return {matched = false}
end
