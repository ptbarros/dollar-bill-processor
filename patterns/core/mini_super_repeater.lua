--[[
Pattern: MINI_SUPER_REPEATER
Description: Alternating start pattern (12121234)
Tier: 4
Examples: ["12121234", "89898912", "80808093"]
Odds: 1 in 4,938
Price: $4-$20
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Pattern: ABABAB.. where first 6 positions are ABABAB
    local a = digits:sub(1, 1)
    local b = digits:sub(2, 2)

    -- Check ABABAB pattern in first 6 positions
    if digits:sub(3, 3) ~= a or digits:sub(4, 4) ~= b or
       digits:sub(5, 5) ~= a or digits:sub(6, 6) ~= b then
        return {matched = false}
    end

    -- A and B must be different
    if a == b then
        return {matched = false}
    end

    return {
        matched = true,
        highlights = {
            highlight({0, 2, 4}, "magenta", "A"),
            highlight({1, 3, 5}, "coral", "B"),
            highlight({6, 7}, "gray", "tail")
        },
        connectors = {
            connector(0, 2, "magenta", "line"),
            connector(2, 4, "magenta", "line"),
            connector(1, 3, "coral", "line"),
            connector(3, 5, "coral", "line")
        },
        message = "Mini super repeater: " .. a .. b .. " x 3 + tail"
    }
end
