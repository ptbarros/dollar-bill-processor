--[[
Pattern: CS_SINGLE_BOOKEND
DisplayName: CS-Single Bookend
Description: The digits in positions 1 and 8 are identical. e.g., M 7xxxxxx7 M.
BookRef: CS-950
Tier: 8
Examples: ["71234567", "31987653", "00000000"]
Odds: 1 in 10,000,000
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    if not is_bookended(d, 1) then
        return {matched = false}
    end

    local outer = d:sub(1, 1)

    return {
        matched = true,
        highlights = {
            {positions = {0, 7}, color = "orange"}
        },
        connectors = {
            {from = 0, to = 7, color = "orange", style = "arc"}
        },
        message = outer .. " bookends the serial (CS-Single Bookend)"
    }
end
