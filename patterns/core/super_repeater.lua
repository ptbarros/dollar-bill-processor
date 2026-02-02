--[[
Pattern: SUPER_REPEATER
Description: 2-digit pattern repeated 4 times (ABABABAB)
Tier: 2
Examples: ["12121212", "78787878", "39393939"]
Odds: 1 in 1,185,185
Price: $80-$900
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check ABABABAB pattern
    if not is_super_repeater(digits) then
        return {matched = false}
    end

    local a = digits:sub(1, 1)
    local b = digits:sub(2, 2)

    -- A and B must be different
    if a == b then
        return {matched = false}  -- That's a solid
    end

    return {
        matched = true,
        highlights = {
            highlight({0, 2, 4, 6}, "magenta", "A digits"),
            highlight({1, 3, 5, 7}, "coral", "B digits")
        },
        connectors = {
            connector(0, 2, "magenta", "line"),
            connector(2, 4, "magenta", "line"),
            connector(4, 6, "magenta", "line")
        },
        message = "Super repeater: " .. a .. b .. " x 4"
    }
end
