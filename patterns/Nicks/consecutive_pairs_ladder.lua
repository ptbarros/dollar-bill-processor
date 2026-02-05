--[[
Pattern: CONSECUTIVE_PAIRS_LADDER
DisplayName: 3 Consecutive Pairs Ladder
Description: At least 3 consecutive identical pairs in ladder order (AABBCC)
Tier: 4
Examples: ["11223344", "22334455", "44332211", "55443322"]
--]]

function match(ctx)
    local s = ctx.digits

    -- Split into pairs
    local pairs = {}
    for i = 1, 8, 2 do
        table.insert(pairs, s:sub(i, i + 1))
    end

    -- Check if each pair has identical digits
    local identical_pairs = {}
    for i, pair in ipairs(pairs) do
        if pair:sub(1, 1) == pair:sub(2, 2) then
            table.insert(identical_pairs, {index = i, digit = pair:sub(1, 1)})
        end
    end

    if #identical_pairs < 3 then
        return {matched = false}
    end

    -- Check if 3+ consecutive pairs form a ladder
    for start = 1, #identical_pairs - 2 do
        local is_asc = true
        local is_desc = true

        for i = start, start + 1 do
            if i + 1 <= #identical_pairs then
                local curr = tonumber(identical_pairs[i].digit)
                local next = tonumber(identical_pairs[i + 1].digit)

                -- Check consecutive indices
                if identical_pairs[i + 1].index - identical_pairs[i].index ~= 1 then
                    is_asc = false
                    is_desc = false
                    break
                end

                if next - curr ~= 1 then is_asc = false end
                if curr - next ~= 1 then is_desc = false end
            end
        end

        if is_asc or is_desc then
            local positions = {}
            for i = start, start + 2 do
                local idx = identical_pairs[i].index
                table.insert(positions, (idx - 1) * 2)
                table.insert(positions, (idx - 1) * 2 + 1)
            end

            local direction = is_asc and "ascending" or "descending"
            return {
                matched = true,
                message = "3 consecutive pairs " .. direction .. " ladder",
                highlights = {{positions = positions, color = is_asc and "lime" or "cyan"}}
            }
        end
    end

    return {matched = false}
end
