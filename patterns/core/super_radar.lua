--[[
Pattern: SUPER_RADAR
Description: A single repeated digit six times, wrapped in a matching pair — one digit at each end, six of another digit between (e.g. 10000001, 58888885)
Tier: 2
Examples: ["10000001", "42222224", "91111119"]
Odds: 1 in 1,103,448 (87 per 96M)
Price: $80-$2,500
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- First and last must be same
    local bookend = digits:sub(1, 1)
    if digits:sub(8, 8) ~= bookend then
        return {matched = false}
    end

    -- All interior digits (2-7) must be same
    local interior = digits:sub(2, 2)
    for i = 3, 7 do
        if digits:sub(i, i) ~= interior then
            return {matched = false}
        end
    end

    -- Bookend and interior must be different
    if bookend == interior then
        return {matched = false}  -- That's a solid
    end

    return {
        matched = true,
        highlights = {
            highlight({0, 7}, "orange", "bookends"),
            highlight({1, 2, 3, 4, 5, 6}, "gold", "interior")
        },
        connectors = {
            connector(0, 7, "orange", "arc")
        },
        message = "Super radar: " .. bookend .. " - 6x" .. interior .. " - " .. bookend
    }
end
