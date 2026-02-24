--[[
Pattern: CS_ROTATOR
DisplayName: CS-Rotator
Description: Serial reads the same when rotated 180°. All digits must be flip-valid (0, 1, 6, 8, 9) and the reversed+mapped version equals the original.
BookRef: CS-1090
Tier: 2
Examples: ["10811801", "96100196", "18188181"]
Price: $5-$500
--]]

function match(ctx)
    local d = ctx.digits

    -- All digits must be flip-valid: 0, 1, 6, 8, 9
    if not all_flip_valid(d) then
        return {matched = false}
    end

    -- Flipped (rotated 180°) must equal the original
    local flipped = flip_string(d)
    if flipped ~= d then
        return {matched = false}
    end

    -- Exclude tetradic (only 0/1/8 + palindrome) — that's a higher pattern
    -- We still match; let the tier system handle stacking

    local positions = {}
    for i = 0, 7 do
        table.insert(positions, i)
    end

    return {
        matched = true,
        highlights = {{positions = positions, color = "purple"}},
        connectors = {
            {from = 0, to = 7, color = "purple", style = "arc"},
            {from = 1, to = 6, color = "purple", style = "arc"},
        },
        message = "CS-Rotator: reads same upside-down (CS-1090)"
    }
end
