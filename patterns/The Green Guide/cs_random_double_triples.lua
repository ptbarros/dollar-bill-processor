--[[
Pattern: CS_RANDOM_DOUBLE_TRIPLES
DisplayName: CS-Random Double Triples
Description: Two distinct digits each appearing 3+ times in any positions (scattered). e.g., M 757755xx M.
BookRef: CS-160
Tier: 6
Examples: ["75775511", "12312321", "77755500"]
Odds: 1 in 24
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    local counts = count_digits(d)

    -- Find digits with 3+ occurrences, ordered 0-9
    local triple_digits = {}
    for n = 0, 9 do
        local s = tostring(n)
        if (counts[s] or 0) >= 3 then
            table.insert(triple_digits, s)
        end
    end

    if #triple_digits < 2 then
        return {matched = false}
    end

    local colors = {"gold", "coral", "cyan", "lime", "orange", "magenta"}
    local highlights = {}
    for i, digit in ipairs(triple_digits) do
        local color = colors[((i - 1) % #colors) + 1]
        table.insert(highlights, {
            positions = find_digit_positions(d, digit),
            color = color
        })
    end

    return {
        matched = true,
        highlights = highlights,
        message = #triple_digits .. " digits × 3+ occurrences (CS-Random Double Triples)"
    }
end
