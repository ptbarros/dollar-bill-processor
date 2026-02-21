--[[
Pattern: CS_MILLION_NOTE
DisplayName: CS-Million Note
Description: Serial ends in 6 or 7 trailing zeros (the last 6-7 digits are all zeros). e.g., M x0000000 M or M xx000000 M.
BookRef: CS-1990
Tier: 3
Examples: ["10000000", "20000000", "23000000"]
Odds: 1 in ~111
Price: $0-$500+
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Count trailing zeros from the end
    local trailing_zeros = 0
    for i = 8, 1, -1 do
        if d:sub(i, i) == "0" then
            trailing_zeros = trailing_zeros + 1
        else
            break
        end
    end

    -- Must have 6 or 7 trailing zeros
    if trailing_zeros < 6 then return {matched = false} end

    -- Must not be all zeros (that's SOLID)
    if trailing_zeros == 8 then return {matched = false} end

    -- Verify all non-trailing-zero positions have no zeros mixed in (pure trailing)
    for i = 1, 8 - trailing_zeros do
        if d:sub(i, i) == "0" then
            return {matched = false}  -- zero before the trailing block
        end
    end

    local zero_positions = {}
    for i = 8 - trailing_zeros, 7 do
        table.insert(zero_positions, i)
    end

    local label = trailing_zeros == 7 and "x0000000" or "xx000000"

    return {
        matched = true,
        highlights = {
            {positions = zero_positions, color = "cyan"}
        },
        message = trailing_zeros .. " trailing zeros (" .. label .. ") — CS-Million Note"
    }
end
