--[[
Pattern: ASCENDING_LADDERED_RADAR
DisplayName: Radar Ladder
Description: Reads the same forwards and backwards, and the first four digits step straight up or straight down (e.g. 4567·7654).
Tier: 1
Odds: 1 in 7,384,615 (13 per 96M)
Examples: ["45677654", "76544567", "12344321", "43211234"]
Price: $5-$100
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Must be a palindrome (radar)
    if not is_palindrome(d) then return {matched = false} end

    -- First 4 digits must form a ladder, either direction (Ed review: merged the
    -- old Ascending and Descending Laddered Radars into one).
    local first = d:sub(1, 4)
    local up = is_ascending(first)
    if not up and not is_descending(first) then return {matched = false} end

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
        message = "Laddered Radar: " .. (up and "ascending" or "descending") .. " first half + palindrome"
    }
end
