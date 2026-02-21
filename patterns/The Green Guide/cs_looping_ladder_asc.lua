--[[
Pattern: CS_LOOPING_LADDER_ASC
DisplayName: CS-Ascending Looping Ladder
Description: An ascending 8-digit ladder that wraps around modulo 10. e.g., 56781234 (5-6-7-8-1-2-3-4... wait that skips 9→0). Actually: consecutive mod 10, like 7890 1234 or 5678 9012.
BookRef: CS-1190
Tier: 1
Examples: ["56789012", "67890123", "78901234"]
Odds: 1 in 6,944,444
Price: $250-$1,000
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Must NOT be a standard ascending ladder (that's CS-1170)
    if is_ascending(d) then
        return {matched = false}
    end

    -- Check if digits form a modulo-10 ascending sequence
    -- i.e., each digit = (first + offset) % 10
    local first = tonumber(d:sub(1, 1))
    local is_loop_asc = true
    for i = 2, 8 do
        local expected = (first + (i - 1)) % 10
        if tonumber(d:sub(i, i)) ~= expected then
            is_loop_asc = false
            break
        end
    end

    if not is_loop_asc then
        return {matched = false}
    end

    -- Must start at 2 or higher (otherwise it would be caught by CS-1170 or start at 0)
    -- But we already excluded is_ascending (which only catches 0123... 1234... etc without wrap)
    -- Verify there's actually a wrap: max digit > min digit somewhere in sequence
    local wrap_found = false
    for i = 1, 7 do
        local curr = tonumber(d:sub(i, i))
        local nxt = tonumber(d:sub(i + 1, i + 1))
        if nxt < curr then
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
        table.insert(connectors, {from = i, to = i + 1, color = "lime", style = "line"})
    end

    return {
        matched = true,
        highlights = {{positions = positions, color = "lime"}},
        connectors = connectors,
        message = "Ascending looping ladder starting at " .. d:sub(1,1) .. " (CS-1190)"
    }
end
