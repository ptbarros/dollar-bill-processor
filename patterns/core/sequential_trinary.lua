--[[
Pattern: SEQUENTIAL_TRINARY
Description: Trinary with 3 sequential digits
Tier: 4
Examples: ["12121212", "23232323", "34343434"]
Odds: 1 in 2,090
Price: $5-$25
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    -- Must be trinary (exactly 3 unique digits)
    if unique_count(digits) ~= 3 then
        return {matched = false}
    end

    -- Get the three unique digits and check if they're consecutive
    local unique = get_unique_digits(digits)
    local d1 = tonumber(unique:sub(1, 1))
    local d2 = tonumber(unique:sub(2, 2))
    local d3 = tonumber(unique:sub(3, 3))

    -- Check if they form a sequence
    if not ((d2 == d1 + 1 and d3 == d2 + 1) or (d2 == d1 - 1 and d3 == d2 - 1)) then
        return {matched = false}
    end

    local highlights = {
        highlight(find_digit_positions(digits, unique:sub(1, 1)), "lime", "digit 1"),
        highlight(find_digit_positions(digits, unique:sub(2, 2)), "teal", "digit 2"),
        highlight(find_digit_positions(digits, unique:sub(3, 3)), "cyan", "digit 3")
    }

    return {
        matched = true,
        highlights = highlights,
        connectors = {},
        message = "Sequential trinary: " .. unique
    }
end
