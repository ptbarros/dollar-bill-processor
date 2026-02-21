--[[
Pattern: CS_TRI_REPEATED_BOOKEND
DisplayName: CS-Tri Repeated Bookend
Description: First three digits repeat at the end in the same order (e.g., 123xx123). The three bookend digits must not all be the same — that is CS-990 (Tri Matched Bookend).
BookRef: CS-1000
Tier: 7
Examples: ["12300123", "45600456", "78900789"]
Odds: 1 in 720,000
Price: $20-$200
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- First 3 digits must match last 3 digits in same order
    if not is_bookended(d, 3) then
        return {matched = false}
    end

    local b1 = d:sub(1, 1)
    local b2 = d:sub(2, 2)
    local b3 = d:sub(3, 3)

    -- Not all three the same digit — that is the Tri Matched Bookend (CS-990)
    if b1 == b2 and b2 == b3 then
        return {matched = false}
    end

    return {
        matched = true,
        group_boxes = {
            {from = 0, to = 2, color = "orange", thickness = 3},
            {from = 5, to = 7, color = "orange", thickness = 3}
        },
        connectors = {
            {from = 0, to = 7, color = "orange", style = "arc"},
            {from = 1, to = 6, color = "coral",  style = "arc"},
            {from = 2, to = 5, color = "cyan",   style = "arc"}
        },
        message = b1 .. b2 .. b3 .. " tri-repeated at both ends (CS-Tri Repeated Bookend)"
    }
end
