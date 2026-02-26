--[[
Pattern: CS_TRUE_BINARY_ROTATOR
DisplayName: CS-True Binary Rotator
Description: Rotator using exactly 2 unique digits, restricted to {0,1} only. Rotation equals reversal equals palindrome for these digits.
BookRef: CS-1110
Tier: 3
Examples: ["01000010", "10000001", "01100110"]
Price: $5-$100
--]]

function match(ctx)
    local d = ctx.digits

    -- Must use only digits 0 and 1
    if not only_digits(d, "01") then return {matched = false} end

    -- Exactly 2 unique digits
    if unique_count(d) ~= 2 then return {matched = false} end

    -- For {0,1} only: flip_string is just reversal, so rotator == palindrome
    -- Still verify the rotator property explicitly
    if not all_flip_valid(d) then return {matched = false} end
    if flip_string(d) ~= d then return {matched = false} end

    local positions = {0, 1, 2, 3, 4, 5, 6, 7}

    return {
        matched = true,
        highlights = {{positions = positions, color = "purple"}},
        connectors = {
            {from = 0, to = 7, color = "purple", style = "arc"},
            {from = 1, to = 6, color = "purple", style = "arc"},
        },
        message = "CS-True Binary Rotator: {0,1} rotator (CS-1110)"
    }
end
