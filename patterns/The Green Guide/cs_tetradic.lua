--[[
Pattern: CS_TETRADIC
DisplayName: CS-Tetradic
Description: Only digits 0, 1, 8; is a palindrome; AND reads the same upside-down (flipped 180°). Only ~2 exist per run.
BookRef: CS-1160
Tier: 1
Examples: ["08111180", "10111101", "10011001"]
Price: $50-$1,000
--]]

function match(ctx)
    local d = ctx.digits

    -- All digits must be 0, 1, or 8 (the only digits that look the same upside-down)
    if not only_digits(d, "018") then
        return {matched = false}
    end

    -- Must be a palindrome (CS-Full Radar)
    if not is_palindrome(d) then
        return {matched = false}
    end

    -- Must read the same when rotated 180° (flip_string reverses and maps 6↔9)
    local flipped = flip_string(d)
    if flipped ~= d then
        return {matched = false}
    end

    local positions = {}
    for i = 0, 7 do
        table.insert(positions, i)
    end

    return {
        matched = true,
        highlights = {{positions = positions, color = "gold"}},
        connectors = {
            {from = 0, to = 7, color = "purple", style = "arc"},
            {from = 1, to = 6, color = "purple", style = "arc"},
            {from = 2, to = 5, color = "purple", style = "arc"},
            {from = 3, to = 4, color = "purple", style = "arc"},
        },
        message = "CS-Tetradic: reads same in all 4 orientations (CS-1160)"
    }
end
