--[[
Pattern: CS_TRINARY_ROTATOR
DisplayName: CS-Trinary Rotator
Description: Rotator using exactly 3 unique flip-valid digits. Valid sets include {0,6,9}, {1,6,9}, {8,6,9}, or {0,1,8}.
BookRef: CS-1130
Tier: 4
Examples: ["06900690", "16911691", "10800801"]
Price: $5-$50
--]]

function match(ctx)
    local d = ctx.digits

    -- Must be flip-valid and rotator
    if not all_flip_valid(d) then return {matched = false} end
    if flip_string(d) ~= d then return {matched = false} end

    -- Exactly 3 unique digits
    if unique_count(d) ~= 3 then return {matched = false} end

    local positions = {0, 1, 2, 3, 4, 5, 6, 7}

    return {
        matched = true,
        highlights = {{positions = positions, color = "purple"}},
        connectors = {
            {from = 0, to = 7, color = "purple", style = "arc"},
            {from = 1, to = 6, color = "purple", style = "arc"},
        },
        message = "CS-Trinary Rotator: 3-digit rotator (CS-1130)"
    }
end
