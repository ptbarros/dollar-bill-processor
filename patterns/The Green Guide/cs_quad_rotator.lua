--[[
Pattern: CS_QUAD_ROTATOR
DisplayName: CS-Quad Rotator
Description: Rotator using exactly 4 unique flip-valid digits. Must include {6,9} plus 2 from {0,1,8}.
BookRef: CS-1140
Tier: 5
Examples: ["01669910", "08696980", "18696981"]
Price: $5-$20
--]]

function match(ctx)
    local d = ctx.digits

    -- Must be flip-valid and rotator
    if not all_flip_valid(d) then return {matched = false} end
    if flip_string(d) ~= d then return {matched = false} end

    -- Exactly 4 unique digits
    if unique_count(d) ~= 4 then return {matched = false} end

    local positions = {0, 1, 2, 3, 4, 5, 6, 7}

    return {
        matched = true,
        highlights = {{positions = positions, color = "purple"}},
        connectors = {
            {from = 0, to = 7, color = "purple", style = "arc"},
            {from = 1, to = 6, color = "purple", style = "arc"},
        },
        message = "CS-Quad Rotator: 4-digit rotator (CS-1140)"
    }
end
