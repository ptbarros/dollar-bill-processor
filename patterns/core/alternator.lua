--[[
Pattern: ALTERNATOR
Description: Digits alternate XAXAXAXA or AXAXAXAX
Tier: 4
Examples: ["12121212", "89898989", "37373737"]
Odds: 1 in 823
Price: $3-$10
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check if the string alternates between two values
    if not is_alternating(digits) then
        return {matched = false}
    end

    local a = digits:sub(1, 1)
    local b = digits:sub(2, 2)

    -- Highlight alternating positions
    local a_positions = {0, 2, 4, 6}
    local b_positions = {1, 3, 5, 7}

    return {
        matched = true,
        highlights = {
            highlight(a_positions, "magenta", "A"),
            highlight(b_positions, "coral", "B")
        },
        connectors = {
            connector(0, 2, "magenta", "line"),
            connector(2, 4, "magenta", "line"),
            connector(4, 6, "magenta", "line"),
            connector(1, 3, "coral", "line"),
            connector(3, 5, "coral", "line"),
            connector(5, 7, "coral", "line")
        },
        message = "Alternator: " .. a .. b .. " pattern"
    }
end
