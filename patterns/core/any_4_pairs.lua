--[[
Pattern: ANY_4_PAIRS
DisplayName: 4 Pairs
Description: 4 pairs total (any positions, e.g., AABBCCDD or ABCABCDD)
Tier: 8
Odds: 1 in 147 (655,103 per 96M)
Examples: ["11223344", "12123434", "11112222", "00001111"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Positions (0-indexed) of each digit value, in left-to-right order.
    local pos = {}
    for i = 1, 8 do
        local d = s:sub(i, i)
        pos[d] = pos[d] or {}
        table.insert(pos[d], i - 1)
    end

    -- Chunk each digit's occurrences into pairs (2 = 1 pair), digits in value
    -- order for a deterministic colour assignment. With exactly 4 pairs across
    -- 8 digits every position lands in a pair (no leftover).
    local digit_vals = {}
    for d, _ in pairs(pos) do table.insert(digit_vals, d) end
    table.sort(digit_vals)

    local pair_list = {}  -- {p1, p2} per pair
    for _, d in ipairs(digit_vals) do
        local ps = pos[d]
        local i = 1
        while i + 1 <= #ps do
            table.insert(pair_list, {ps[i], ps[i + 1]})
            i = i + 2
        end
    end

    if #pair_list == 4 then
        -- One distinct colour per pair so the four pairs read apart on the
        -- overlay (the rotation maps first-seen names onto blue/orange/…).
        local colors = {"blue", "orange", "magenta", "red"}
        local highlights = {}
        for i, pr in ipairs(pair_list) do
            table.insert(highlights, {positions = {pr[1], pr[2]}, color = colors[i]})
        end
        return {
            matched = true,
            message = "4 pairs total",
            highlights = highlights
        }
    end

    return {matched = false}
end
