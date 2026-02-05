--[[
Pattern: TRIPLE_QUAD
DisplayName: Triple & Quad
Description: One digit appears 4 times, another appears 3 times
Tier: 4
Examples: ["11112223", "00001112", "99998887", "55554443"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Count occurrences
    local counts = {}
    for i = 1, 8 do
        local d = s:sub(i, i)
        counts[d] = (counts[d] or 0) + 1
    end

    local has_quad = false
    local has_triple = false
    local quad_digit, triple_digit

    for digit, count in pairs(counts) do
        if count == 4 then
            has_quad = true
            quad_digit = digit
        elseif count == 3 then
            has_triple = true
            triple_digit = digit
        end
    end

    if has_quad and has_triple then
        -- Find positions
        local quad_pos = {}
        local triple_pos = {}
        for i = 1, 8 do
            if s:sub(i, i) == quad_digit then
                table.insert(quad_pos, i - 1)
            elseif s:sub(i, i) == triple_digit then
                table.insert(triple_pos, i - 1)
            end
        end

        return {
            matched = true,
            message = "Triple & quad: " .. quad_digit .. "×4 + " .. triple_digit .. "×3",
            highlights = {
                {positions = quad_pos, color = "orange"},
                {positions = triple_pos, color = "cyan"}
            }
        }
    end

    return {matched = false}
end
