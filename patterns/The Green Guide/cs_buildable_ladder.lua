--[[
Pattern: CS_BUILDABLE_LADDER
DisplayName: CS-Buildable Ladder
Description: A scattered ladder with some numbers missing — k of 8 digits form a consecutive mod-10 set in scrambled order. At least one digit must be out of positional order (not a broken/full ladder). Minimum k=3; k=8 is CS-Scattered Ladder (excluded).
BookRef: CS-1250
Tier: 8
Examples: ["34576000", "34657000", "34675000"]
Odds: 1 in 5
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Collect digit values present in serial
    local present = {}
    for i = 1, 8 do
        present[tonumber(d:sub(i, i))] = true
    end

    -- For each starting value, find longest consecutive mod-10 run where all values are present
    local best_k = 0
    local best_start = 0

    for s = 0, 9 do
        local k = 0
        for j = 0, 9 do
            local val = (s + j) % 10
            if present[val] then
                k = k + 1
            else
                break
            end
        end
        if k > best_k then
            best_k = k
            best_start = s
        end
    end

    -- k=8 is CS-Scattered Ladder, excluded; minimum reportable k=3
    if best_k < 3 or best_k >= 8 then return {matched = false} end

    -- Build the set of consecutive digit values
    local consec_set = {}
    for j = 0, best_k - 1 do
        consec_set[(best_start + j) % 10] = true
    end

    -- Find first-occurrence positions of each consecutive digit value
    local first_positions = {}  -- {value, position} pairs
    for j = 0, best_k - 1 do
        local val = (best_start + j) % 10
        for i = 1, 8 do
            if tonumber(d:sub(i, i)) == val then
                table.insert(first_positions, {value = val, pos = i})
                break
            end
        end
    end

    -- Sort by position to get the values in their serial order
    table.sort(first_positions, function(a, b) return a.pos < b.pos end)

    -- Check if the values in position-order form a strictly ascending or descending
    -- mod-10 sequence — if so, it's a broken/full ladder, not buildable
    local values_in_order = {}
    for _, fp in ipairs(first_positions) do
        table.insert(values_in_order, fp.value)
    end

    local is_asc = true
    local is_desc = true
    for i = 1, #values_in_order - 1 do
        local curr = values_in_order[i]
        local next_val = values_in_order[i + 1]
        if next_val ~= (curr + 1) % 10 then is_asc = false end
        if next_val ~= (curr - 1 + 10) % 10 then is_desc = false end
    end

    if is_asc or is_desc then return {matched = false} end

    -- Build visualization: highlight all positions containing consecutive set digits
    local highlight_positions = {}
    for i = 1, 8 do
        local val = tonumber(d:sub(i, i))
        if consec_set[val] then
            table.insert(highlight_positions, i - 1)
        end
    end

    return {
        matched = true,
        highlights = {
            {positions = highlight_positions, color = "lime"}
        },
        message = "Buildable Ladder #" .. best_k .. " (digits " .. best_start .. "-" .. ((best_start + best_k - 1) % 10) .. " scrambled)"
    }
end
