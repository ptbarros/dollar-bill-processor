--[[
Pattern: CS_RANDOM_ZEROS
DisplayName: CS-Random Zeros
Description: Serial contains 1-7 zero digits anywhere. Parent of Leading/Centered/Trailing Zeros. Very common pattern (~57% of serials).
BookRef: CS-1930
Tier: 8
Examples: ["10000000", "00000001", "12345060"]
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Find all zero positions
    local zero_positions = find_digit_positions(d, "0")
    local count = #zero_positions

    -- Must have 1-7 zeros (8 zeros = 00000000, not collectible)
    if count < 1 or count > 7 then
        return {matched = false}
    end

    return {
        matched = true,
        highlights = {
            {positions = zero_positions, color = "cyan"},
        },
        message = count .. " zero" .. (count > 1 and "s" or "") .. " (CS-Random Zeros)"
    }
end
