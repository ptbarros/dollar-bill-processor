--[[
Pattern: BOOKENDED_LADDER
Description: Ladder bookended by matching digits
Tier: 6
Examples: ["77234577", "11234511"]
Odds: 1 in 266,666
Price: $20-$100
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check bookend pattern: XX....XX where first 2 and last 2 match
    if digits:sub(1, 2) ~= digits:sub(7, 8) then
        return {matched = false}
    end

    -- Check that first two are the same digit
    if digits:sub(1, 1) ~= digits:sub(2, 2) then
        return {matched = false}
    end

    local bookend_digit = digits:sub(1, 1)

    -- Look for a ladder of 4+ in the middle (positions 2-5, 0-indexed)
    local middle = digits:sub(3, 6)
    local result = find_ladder_of_length(middle, 4)

    if not result then
        return {matched = false}
    end

    local positions = {}
    for i = 0, result.length - 1 do
        table.insert(positions, 2 + result.start + i)
    end

    return {
        matched = true,
        highlights = {
            highlight({0, 1}, "orange", "bookend"),
            highlight({6, 7}, "orange", "bookend"),
            highlight(positions, "lime", "ladder")
        },
        connectors = {
            connector(0, 7, "orange", "arc"),
            connector(1, 6, "orange", "arc")
        },
        message = "Bookended ladder: " .. bookend_digit .. bookend_digit
    }
end
