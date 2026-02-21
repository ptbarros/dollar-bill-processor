--[[
Pattern: CS_DUAL_BOOKEND
DisplayName: CS-Dual Matched Bookend
Description: The first two digits match the last two digits (same pair at each end). e.g., M 22xxxx22 M.
BookRef: CS-960
Tier: 7
Examples: ["22123422", "44567844", "99000199"]
Odds: 1 in 810,000
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- First 2 digits must equal last 2 digits
    if not is_bookended(d, 2) then
        return {matched = false}
    end

    local outer1 = d:sub(1, 1)
    local outer2 = d:sub(2, 2)

    -- Dual Matched Bookend: both bookend digits are the same (a pair at each end)
    -- e.g., 22xxxx22. Per CS-960, the pair is the same digit at both ends.
    -- CS-980 (Repeater Bookend) is 12xxxx12 (repeated, not necessarily same).
    -- CS-Radar Bookend is 23xxxx32 (mirrored).
    -- CS-960 specifically requires the same pair (AA...AA).
    if outer1 ~= outer2 then
        return {matched = false}
    end

    return {
        matched = true,
        group_boxes = {
            {from = 0, to = 1, color = "orange", thickness = 3},
            {from = 6, to = 7, color = "orange", thickness = 3}
        },
        connectors = {
            {from = 0, to = 7, color = "orange", style = "arc"},
            {from = 1, to = 6, color = "coral", style = "arc"}
        },
        message = outer1 .. outer1 .. " pair bookends serial (CS-Dual Matched Bookend)"
    }
end
