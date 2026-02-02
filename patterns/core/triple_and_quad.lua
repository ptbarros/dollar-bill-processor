--[[
Pattern: TRIPLE_AND_QUAD
Description: Triple + Quad combination
Tier: 3
Examples: ["11122222", "33334445", "00011111"]
Odds: 1 in 24,691
Price: $20-$100+
--]]

function match(ctx)
    local digits = ctx.digits
    if #digits ~= 8 then
        return {matched = false}
    end

    local counts = count_digits(digits)

    -- Find digits with count 4+ and count 3
    local quad_digit = nil
    local triple_digit = nil

    for d, c in pairs(counts) do
        if c >= 4 and not quad_digit then
            quad_digit = d
        elseif c == 3 then
            triple_digit = d
        end
    end

    if not quad_digit or not triple_digit then
        return {matched = false}
    end

    local quad_pos = find_digit_positions(digits, quad_digit)
    local triple_pos = find_digit_positions(digits, triple_digit)

    return {
        matched = true,
        highlights = {
            highlight(quad_pos, "gold", "quad"),
            highlight(triple_pos, "coral", "triple")
        },
        connectors = {},
        message = "Triple + Quad: 3x" .. triple_digit .. " + 4x" .. quad_digit
    }
end
