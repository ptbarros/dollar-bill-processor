--[[
Pattern: SERIAL_UNDER_10
Description: Serial number 00000001-00000009
Tier: 1
Examples: ["00000001", "00000003", "00000009"]
Odds: 1 in 10,666,667
Price: $300-$25,000
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Check first 7 digits are 0
    if not starts_with(digits, "0000000") then
        return {matched = false}
    end

    -- Last digit must be 1-9
    local last = digits:sub(8, 8)
    if last == "0" then
        return {matched = false}
    end

    return {
        matched = true,
        -- Box only the non-zero serial digit; leading zeros drawn plain (Ed review).
        highlights = {
            highlight({7}, "blue", "serial number")
        },
        connectors = {},
        message = "Serial #" .. last
    }
end
