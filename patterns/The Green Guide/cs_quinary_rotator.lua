--[[
Pattern: CS_QUINARY_ROTATOR
DisplayName: CS-Quinary Rotator
Description: Rotator using all 5 flip-valid digits {0,1,6,8,9}.
BookRef: CS-1150
Tier: 5
Examples: ["16800891", "01988610"]
Price: $5-$20
--]]

function match(ctx)
    local d = ctx.digits

    -- Must be flip-valid and rotator
    if not all_flip_valid(d) then return {matched = false} end
    if flip_string(d) ~= d then return {matched = false} end

    -- Exactly 5 unique digits (all flip-valid digits used)
    if unique_count(d) ~= 5 then return {matched = false} end

    local positions = {0, 1, 2, 3, 4, 5, 6, 7}

    return {
        matched = true,
        highlights = {{positions = positions, color = "purple"}},
        connectors = {
            {from = 0, to = 7, color = "purple", style = "arc"},
            {from = 1, to = 6, color = "purple", style = "arc"},
        },
        message = "CS-Quinary Rotator: all 5 flip digits (CS-1150)"
    }
end
