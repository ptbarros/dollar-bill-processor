--[[
Pattern: TRUE_BINARY_ROTATOR
DisplayName: True Binary Rotator
Description: Built from only 0s and 1s, both present, and it reads the same turned upside-down (e.g. 0100·0010).
Tier: 1
Flippable: true
Odds: 1 in 6,857,143 (14 per 96M)
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

    -- Colour the two digit values distinctly (0 vs 1); keep the rotation arcs.
    local highlights = {}
    for i = 0, 7 do
        local ch = d:sub(i + 1, i + 1)
        table.insert(highlights, {positions = {i}, color = (ch == "0") and "blue" or "orange"})
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = {
            {from = 0, to = 7, color = "magenta", style = "arc"},
            {from = 1, to = 6, color = "magenta", style = "arc"},
        },
        message = "True Binary Rotator: {0,1} rotator"
    }
end
