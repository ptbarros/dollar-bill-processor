--[[
Pattern: CS_TRAILING_ZEROS
DisplayName: CS-Trailing Zeros
Description: Serial ends with one or more zeros, all zeros contiguous at the end. e.g., M xxxxxxx0 M or M xxxxx000 M. The more trailing zeros, the rarer.
BookRef: CS-1960
Tier: 7
Examples: ["12345600", "12345000", "12340000"]
Odds: 1 in 10,000,000
Price: $0-$500+
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Last digit must be 0
    if d:sub(8, 8) ~= "0" then
        return {matched = false}
    end

    -- Count contiguous trailing zeros
    local trailing_zeros = 0
    for i = 8, 1, -1 do
        if d:sub(i, i) == "0" then
            trailing_zeros = trailing_zeros + 1
        else
            break
        end
    end

    -- Must not be all zeros (that would be SOLID)
    if trailing_zeros == 8 then
        return {matched = false}
    end

    -- Verify no zeros appear before the trailing block
    for i = 1, 8 - trailing_zeros do
        if d:sub(i, i) == "0" then
            return {matched = false}
        end
    end

    -- Highlight trailing zeros
    local zero_positions = {}
    for i = 8 - trailing_zeros, 7 do
        table.insert(zero_positions, i)
    end

    return {
        matched = true,
        highlights = {
            {positions = zero_positions, color = "cyan"}
        },
        message = trailing_zeros .. " trailing zero(s) (CS-1960)"
    }
end
