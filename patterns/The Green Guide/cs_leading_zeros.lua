--[[
Pattern: CS_LEADING_ZEROS
DisplayName: CS-Leading Zeros
Description: Serial starts with one or more zeros, all zeros contiguous at the front. e.g., M 00000xxx M. The more leading zeros, the rarer.
BookRef: CS-1940
Tier: 7
Examples: ["00012345", "00064185", "01234567"]
Odds: 1 in 10,000,000
Price: $0-$500+
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- First digit must be 0
    if d:sub(1, 1) ~= "0" then
        return {matched = false}
    end

    -- Count contiguous leading zeros
    local leading_zeros = 0
    for i = 1, 8 do
        if d:sub(i, i) == "0" then
            leading_zeros = leading_zeros + 1
        else
            break
        end
    end

    -- Must not be all zeros (that would be SOLID)
    if leading_zeros == 8 then
        return {matched = false}
    end

    -- Verify no zeros appear after the leading block
    -- (zeros must only be contiguous at the start)
    for i = leading_zeros + 1, 8 do
        if d:sub(i, i) == "0" then
            return {matched = false}
        end
    end

    -- Highlight leading zeros
    local zero_positions = {}
    for i = 0, leading_zeros - 1 do
        table.insert(zero_positions, i)
    end

    return {
        matched = true,
        highlights = {
            {positions = zero_positions, color = "cyan"}
        },
        message = leading_zeros .. " leading zero(s) (CS-1940)"
    }
end
