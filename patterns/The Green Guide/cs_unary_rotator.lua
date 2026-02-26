--[[
Pattern: CS_UNARY_ROTATOR
DisplayName: CS-Unary Rotator
Description: Rotator using only 1 unique digit. Only 11111111 and 88888888 qualify (00000000 excluded as non-collectible).
BookRef: CS-1100
Tier: 1
Examples: ["11111111", "88888888"]
Price: $5-$500
--]]

function match(ctx)
    local d = ctx.digits

    -- Must be flip-valid and rotator
    if not all_flip_valid(d) then return {matched = false} end
    if flip_string(d) ~= d then return {matched = false} end

    -- Exactly 1 unique digit
    if unique_count(d) ~= 1 then return {matched = false} end

    -- Exclude 00000000
    if d == "00000000" then return {matched = false} end

    local positions = {0, 1, 2, 3, 4, 5, 6, 7}

    return {
        matched = true,
        highlights = {{positions = positions, color = "purple"}},
        connectors = {
            {from = 0, to = 7, color = "purple", style = "arc"},
            {from = 1, to = 6, color = "purple", style = "arc"},
        },
        message = "CS-Unary Rotator: single-digit rotator (CS-1100)"
    }
end
