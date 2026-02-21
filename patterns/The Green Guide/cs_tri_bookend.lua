--[[
Pattern: CS_TRI_BOOKEND
DisplayName: CS-Tri Matched Bookend
Description: The first three digits match the last three digits. e.g., M 333xx333 M.
BookRef: CS-990
Tier: 6
Examples: ["33300333", "77712777", "12312123"]
Odds: 1 in 720
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- First 3 digits must equal last 3 digits
    if not is_bookended(d, 3) then
        return {matched = false}
    end

    local b1 = d:sub(1, 1)
    local b2 = d:sub(2, 2)
    local b3 = d:sub(3, 3)

    return {
        matched = true,
        group_boxes = {
            {from = 0, to = 2, color = "orange", thickness = 3},
            {from = 5, to = 7, color = "orange", thickness = 3}
        },
        connectors = {
            {from = 0, to = 7, color = "orange", style = "arc"},
            {from = 1, to = 6, color = "coral", style = "arc"},
            {from = 2, to = 5, color = "cyan", style = "arc"}
        },
        message = b1 .. b2 .. b3 .. " tri-bookends serial (CS-Tri Matched Bookend)"
    }
end
