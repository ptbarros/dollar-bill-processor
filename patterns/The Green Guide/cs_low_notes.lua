--[[
Pattern: CS_LOW_NOTES
DisplayName: CS-Low Notes
Description: Serial number <= 9999 (first four digits are all zeros). The lower the better. e.g., M 0000xxxx M.
BookRef: CS-1980
Tier: 4
Examples: ["00009999", "00000001", "00001234"]
Odds: 1 in 9,000
Price: $5-$15,000+
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- First four digits must all be zeros
    if d:sub(1, 4) ~= "0000" then
        return {matched = false}
    end

    -- Count total leading zeros for message
    local leading_zeros = 0
    for i = 1, 8 do
        if d:sub(i, i) == "0" then
            leading_zeros = leading_zeros + 1
        else
            break
        end
    end

    local n = tonumber(d)

    -- Highlight leading zeros
    local zero_positions = {}
    for i = 0, leading_zeros - 1 do
        table.insert(zero_positions, i)
    end

    return {
        matched = true,
        highlights = {
            {positions = zero_positions, color = "cyan"}
        },
        message = "Serial #" .. n .. " — CS-Low Note (CS-1980)"
    }
end
