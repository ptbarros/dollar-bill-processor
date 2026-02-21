--[[
Pattern: CS_STAND_ALONE_SINGLE
DisplayName: CS-Stand Alone Single
Description: A single non-zero digit surrounded by zeros on all sides. e.g., M 00010000 M or M 00000100 M.
BookRef: CS-1650
Tier: 4
Examples: ["00010000", "00000100", "01000000"]
Odds: 1 in 9
Price: $25-$1,000
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- First and last digits must be zero (surrounded)
    if d:sub(1, 1) ~= "0" or d:sub(8, 8) ~= "0" then
        return {matched = false}
    end

    -- Exactly one non-zero digit, at positions 2-7 (1-indexed), all others zero
    local nonzero_pos = nil
    for i = 1, 8 do
        if d:sub(i, i) ~= "0" then
            if nonzero_pos ~= nil then
                return {matched = false}  -- More than one non-zero digit
            end
            nonzero_pos = i
        end
    end

    if nonzero_pos == nil then
        return {matched = false}  -- All zeros = SOLID
    end

    -- Must be at positions 2-7 (already guaranteed by first/last zero check + single nonzero)
    local digit = d:sub(nonzero_pos, nonzero_pos)
    local pos0 = nonzero_pos - 1  -- 0-indexed

    return {
        matched = true,
        highlights = {
            {positions = {pos0}, color = "gold"}
        },
        message = digit .. " stand-alone at position " .. nonzero_pos .. " (CS-Stand Alone Single)"
    }
end
