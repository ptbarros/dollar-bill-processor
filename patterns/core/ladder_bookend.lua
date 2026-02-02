--[[
Pattern: LADDER_BOOKEND
Description: Ladder sequence bookended by same digits
Tier: 3
Examples: ["12345621", "77234577", "11234511"]
Odds: 1 in 41,666
Price: $10-$50
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check if first and last digits match
    if digits:sub(1, 1) ~= digits:sub(8, 8) then
        return {matched = false}
    end

    -- Look for a ladder in the middle portion
    local bookend = digits:sub(1, 1)

    -- Check positions 1-6 (0-indexed) or 2-7 for ladder of 4+
    for start = 1, 3 do
        local result = find_ladder_of_length(digits:sub(start + 1), 4)
        if result then
            local positions = {}
            for i = 0, result.length - 1 do
                table.insert(positions, start + result.start + i)
            end

            return {
                matched = true,
                highlights = {
                    highlight({0}, "orange", "bookend"),
                    highlight({7}, "orange", "bookend"),
                    highlight(positions, "lime", "ladder")
                },
                connectors = {
                    connector(0, 7, "orange", "arc")
                },
                message = "Ladder with " .. bookend .. " bookends"
            }
        end
    end

    return {matched = false}
end
