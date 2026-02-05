--[[
Pattern: NICKS_TRIPLE_DOUBLE_DOUBLE
DisplayName: Triple Dbl Dbl
Description: One triple and two doubles (AAABBCC pattern)
Tier: 5
Examples: ["11122334", "00011223", "99988776", "55544332"]
--]]

function match(ctx)
    local s = ctx.digits

    local i = 1
    local triples = 0
    local doubles = 0

    while i <= 8 do
        -- Check for triple
        if i <= 6 and s:sub(i, i) == s:sub(i + 1, i + 1) and s:sub(i, i) == s:sub(i + 2, i + 2) then
            -- Make sure it's not a quad
            if i + 3 > 8 or s:sub(i, i) ~= s:sub(i + 3, i + 3) then
                triples = triples + 1
                i = i + 3
            else
                i = i + 1
            end
        -- Check for double
        elseif i <= 7 and s:sub(i, i) == s:sub(i + 1, i + 1) then
            -- Make sure it's not a triple
            if i + 2 > 8 or s:sub(i, i) ~= s:sub(i + 2, i + 2) then
                doubles = doubles + 1
                i = i + 2
            else
                i = i + 1
            end
        else
            i = i + 1
        end
    end

    if triples == 1 and doubles == 2 then
        return {
            matched = true,
            message = "Triple + double + double",
            highlights = {{positions = {0,1,2,3,4,5,6,7}, color = "purple"}}
        }
    end

    return {matched = false}
end
