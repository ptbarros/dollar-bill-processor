--[[
Pattern: ANY_4_PAIRS
DisplayName: 4 Pairs
Description: 4 pairs total (any positions, e.g., AABBCCDD or ABCABCDD)
Tier: 5
Examples: ["11223344", "12123434", "11112222", "00001111"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Count occurrences
    local counts = {}
    for i = 1, 8 do
        local d = s:sub(i, i)
        counts[d] = (counts[d] or 0) + 1
    end

    -- Count pairs (each 2 of a digit = 1 pair)
    local pairs_count = 0
    for _, count in pairs(counts) do
        pairs_count = pairs_count + math.floor(count / 2)
    end

    if pairs_count == 4 then
        return {
            matched = true,
            message = "4 pairs total",
            highlights = {{positions = {0,1,2,3,4,5,6,7}, color = "gold"}}
        }
    end

    return {matched = false}
end
