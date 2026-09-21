--[[
Pattern: CS_LOOPING_LADDER_ASC
DisplayName: Looping Ladder
Description: Eight digits climbing or dropping by one that wrap around — the straight run rolls off one end and picks back up at the other (e.g. 7812·3456).
BookRef: CS-1190
Tier: 1
Examples: ["78123456", "45678923", "32987654", "18765432"]
Odds: 1 in 721,805 (133 per 96M)
Price: $250-$1,000
--]]

-- Ed review: merged the Ascending and Descending Looping Ladders into one
-- "Looping Ladder" that matches either direction. (CS-1190 / CS-1200.)

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- All 8 digits must be unique (a full 8-of-10 consecutive set, rotated).
    local counts = {}
    for i = 1, 8 do
        local dig = d:sub(i, i)
        counts[dig] = (counts[dig] or 0) + 1
        if counts[dig] > 1 then return {matched = false} end
    end

    -- Try one direction (step = +1 ascending, -1 descending). Returns the wrap
    -- index (0-indexed position of the last digit before the wrap) or nil.
    local function detect(step)
        -- k = start of the consecutive-mod-10 set: the value whose predecessor in
        -- this direction (k - step) is not present.
        local k = nil
        for c = 0, 9 do
            local prev = tostring((c - step + 10) % 10)
            if not counts[prev] then
                local ok = true
                for j = 0, 7 do
                    if not counts[tostring((c + step * j + 100) % 10)] then
                        ok = false
                        break
                    end
                end
                if ok then
                    k = c
                    break
                end
            end
        end
        if k == nil then return nil end

        -- Non-trivial rotation: first digit must not equal k (that is a plain ladder).
        if tonumber(d:sub(1, 1)) == k then return nil end

        local end_val = (k + step * 7 + 100) % 10
        local wrap = nil
        for i = 1, 7 do
            local curr = tonumber(d:sub(i, i))
            local nxt = tonumber(d:sub(i + 1, i + 1))
            local expected = (curr == end_val) and k or (curr + step + 10) % 10
            if nxt ~= expected then return nil end
            if curr == end_val then wrap = i - 1 end
        end
        return wrap
    end

    local dir, wrap = "ascending", detect(1)
    if wrap == nil then
        dir, wrap = "descending", detect(-1)
    end
    if wrap == nil then return {matched = false} end

    -- One box per run split at the wrap, a direction arrow under each.
    local group_boxes = {
        {from = 0, to = wrap, color = "blue", thickness = 3},
        {from = wrap + 1, to = 7, color = "orange", thickness = 3},
    }
    local connectors = {}
    if wrap > 0 then
        table.insert(connectors, {from = 0, to = wrap, color = "blue", style = "arrow"})
    end
    if wrap + 1 < 7 then
        table.insert(connectors, {from = wrap + 1, to = 7, color = "orange", style = "arrow"})
    end

    return {
        matched = true,
        highlights = {},
        connectors = connectors,
        group_boxes = group_boxes,
        message = dir .. " looping ladder (Looping Ladder)"
    }
end
