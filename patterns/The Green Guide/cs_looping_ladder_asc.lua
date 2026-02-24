--[[
Pattern: CS_LOOPING_LADDER_ASC
DisplayName: CS-Ascending Looping Ladder
Description: A cyclic rotation of 8 consecutive mod-10 digits in ascending order, where the sequence does not start at its natural first element (that would be CS-Ascending Ladder). e.g., M 78123456 M (digits 1-8 rotated to start at 7).
BookRef: CS-1190
Tier: 1
Examples: ["78123456", "45678923", "78903456"]
Odds: 1 in 6,944,444
Price: $250-$1,000
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- All 8 digits must be unique
    local counts = {}
    for i = 1, 8 do
        local dig = d:sub(i, i)
        counts[dig] = (counts[dig] or 0) + 1
        if counts[dig] > 1 then return {matched = false} end
    end

    -- Find k: the natural start of the 8-consecutive-mod-10 set.
    -- k is the unique value whose predecessor (k-1 mod 10) is NOT in the set.
    local k = nil
    for candidate = 0, 9 do
        local prev = tostring((candidate - 1 + 10) % 10)
        if not counts[prev] then
            local valid = true
            for j = 0, 7 do
                if not counts[tostring((candidate + j) % 10)] then
                    valid = false
                    break
                end
            end
            if valid then
                k = candidate
                break
            end
        end
    end
    if k == nil then return {matched = false} end

    -- Non-trivial rotation: first digit must NOT equal k (that is CS-Ascending Ladder)
    local first = tonumber(d:sub(1, 1))
    if first == k then return {matched = false} end

    -- Check cyclic ascending order: each consecutive pair advances by +1 within the set,
    -- except after the last element (k+7)%10 which wraps back to k.
    local end_val = (k + 7) % 10
    for i = 1, 7 do
        local curr = tonumber(d:sub(i, i))
        local nxt  = tonumber(d:sub(i + 1, i + 1))
        local expected = (curr == end_val) and k or (curr + 1) % 10
        if nxt ~= expected then return {matched = false} end
    end

    local positions = {}
    for i = 0, 7 do table.insert(positions, i) end

    local connectors = {}
    for i = 0, 6 do
        table.insert(connectors, {from = i, to = i + 1, color = "lime", style = "line"})
    end

    return {
        matched = true,
        highlights = {{positions = positions, color = "lime"}},
        connectors = connectors,
        message = "Ascending looping ladder (k=" .. k .. ", starting at " .. d:sub(1,1) .. ") (CS-1190)"
    }
end
