--[[
Pattern: NEAR_FLIPPER
Description: Readable when flipped upside down (only 0,1,6,8,9)
Tier: 4
Examples: ["18980908", "90898986", "60908818"]
Odds: 1 in 256
Price: $5-$15
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check all digits are flip-valid (0, 1, 6, 8, 9)
    if not all_flip_valid(digits) then
        return {matched = false}
    end

    -- Get the flipped version
    local flipped = flip_string(digits)
    if not flipped then
        return {matched = false}
    end

    -- Near flipper: flips to a valid but different number
    -- It's "near" because it uses flip-valid digits but doesn't read the same
    if flipped == digits then
        return {matched = false}  -- That's a true flipper
    end

    -- Color-code based on how digits transform
    local highlights = {}
    for i = 0, 7 do
        local d = digits:sub(i + 1, i + 1)
        local color
        if d == "0" or d == "8" then
            color = "purple"  -- These stay the same when flipped
        elseif d == "1" then
            color = "blue"    -- 1 stays 1
        else
            color = "magenta" -- 6 and 9 swap
        end
        table.insert(highlights, highlight({i}, color, d))
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = {},
        message = "Near flipper: flips to " .. flipped
    }
end
