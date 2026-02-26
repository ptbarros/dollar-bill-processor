--[[
Pattern: CS_PRIME_NUMBERS
DisplayName: CS-Prime Numbers
Description: The 8-digit serial number, interpreted as an integer, is a prime number.
BookRef: CS-2290
Tier: 8
Examples: ["00055291", "00000002", "99999989"]
Odds: 1 in 20
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    local n = tonumber(d)
    if n < 2 then return {matched = false} end

    -- Trial division primality test
    if n == 2 or n == 3 then
        -- prime
    elseif n % 2 == 0 or n % 3 == 0 then
        return {matched = false}
    else
        local i = 5
        while i * i <= n do
            if n % i == 0 or n % (i + 2) == 0 then
                return {matched = false}
            end
            i = i + 6
        end
    end

    return {
        matched = true,
        highlights = {
            highlight_range(0, 7, "green")
        },
        message = n .. " is prime (CS-Prime Numbers)"
    }
end
