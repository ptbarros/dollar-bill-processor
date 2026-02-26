--[[
Pattern: CS_DESCENDING_LADDERED_RADAR
DisplayName: CS-Descending Laddered Radar
Description: A palindrome (radar) where the first 4 digits form a descending ladder (each -1). e.g., 76544567.
BookRef: CS-1430
Tier: 3
Examples: ["76544567", "43211234", "32100123"]
Price: $5-$100
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Must be a palindrome (radar)
    if not is_palindrome(d) then return {matched = false} end

    -- First 4 digits must form descending ladder
    if not is_descending(d:sub(1, 4)) then return {matched = false} end

    -- Ladder half highlighted in lime, palindrome arcs in purple
    return {
        matched = true,
        highlights = {
            highlight_range(0, 3, "lime"),
            highlight_range(4, 7, "purple"),
        },
        connectors = {
            {from = 0, to = 7, color = "purple", style = "arc"},
            {from = 1, to = 6, color = "purple", style = "arc"},
            {from = 2, to = 5, color = "purple", style = "arc"},
            {from = 3, to = 4, color = "purple", style = "arc"},
        },
        message = "CS-Descending Laddered Radar: descending first half + palindrome (CS-1430)"
    }
end
