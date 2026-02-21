--[[
Pattern: CS_LOOPING_LADDER_DESC
DisplayName: CS-Descending Looping Ladder
Description: A descending 8-digit ladder that wraps around modulo 10. e.g., 43210987 (4-3-2-1-0-9-8-7). Must not start at 9 (that would be a standard descending ladder 98765432).
BookRef: CS-1200
Tier: 1
Examples: ["43210987", "32109876", "21098765"]
Odds: 1 in 6,944,444
Price: $250-$1,000
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Must NOT be a standard descending ladder (that's CS-1180)
    if is_descending(d) then
        return {matched = false}
    end

    -- Check if digits form a modulo-10 descending sequence
    local first = tonumber(d:sub(1, 1))
    local is_loop_desc = true
    for i = 2, 8 do
        local expected = (first - (i - 1)) % 10
        if tonumber(d:sub(i, i)) ~= expected then
            is_loop_desc = false
            break
        end
    end

    if not is_loop_desc then
        return {matched = false}
    end

    -- Must have an actual wrap (a step up from 0 to 9)
    local wrap_found = false
    for i = 1, 7 do
        local curr = tonumber(d:sub(i, i))
        local nxt = tonumber(d:sub(i + 1, i + 1))
        if nxt > curr then
            wrap_found = true
            break
        end
    end

    if not wrap_found then
        return {matched = false}
    end

    local positions = {}
    for i = 0, 7 do
        table.insert(positions, i)
    end

    local connectors = {}
    for i = 0, 6 do
        table.insert(connectors, {from = i, to = i + 1, color = "coral", style = "line"})
    end

    return {
        matched = true,
        highlights = {{positions = positions, color = "coral"}},
        connectors = connectors,
        message = "Descending looping ladder starting at " .. d:sub(1,1) .. " (CS-1200)"
    }
end
