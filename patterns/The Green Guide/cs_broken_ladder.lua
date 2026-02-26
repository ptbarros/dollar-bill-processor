--[[
Pattern: CS_BROKEN_LADDER
DisplayName: CS-Broken Ladder
Description: A consecutive run of 3-7 ascending OR descending digits anywhere in the serial. Parent of CS-Ascending Broken Ladder (CS-1230) and CS-Descending Broken Ladder (CS-1240). Excludes full 8-digit ladders.
BookRef: CS-1220
Tier: 7
Examples: ["12378456", "00054321", "43218765"]
Odds: 1 in 6
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Exclude full 8-digit ascending or descending ladder
    if is_ascending(d) or is_descending(d) then
        return {matched = false}
    end

    -- Find longest ascending run
    local best_asc_start = -1
    local best_asc_len = 0
    local i = 1
    while i <= 8 do
        local run_len = 1
        while i + run_len <= 8 do
            local curr = tonumber(d:sub(i + run_len - 1, i + run_len - 1))
            local nxt = tonumber(d:sub(i + run_len, i + run_len))
            if nxt == curr + 1 then
                run_len = run_len + 1
            else
                break
            end
        end
        if run_len > best_asc_len then
            best_asc_len = run_len
            best_asc_start = i
        end
        i = i + run_len
    end

    -- Find longest descending run
    local best_desc_start = -1
    local best_desc_len = 0
    i = 1
    while i <= 8 do
        local run_len = 1
        while i + run_len <= 8 do
            local curr = tonumber(d:sub(i + run_len - 1, i + run_len - 1))
            local nxt = tonumber(d:sub(i + run_len, i + run_len))
            if nxt == curr - 1 then
                run_len = run_len + 1
            else
                break
            end
        end
        if run_len > best_desc_len then
            best_desc_len = run_len
            best_desc_start = i
        end
        i = i + run_len
    end

    -- Pick the longer run
    local best_start, best_len, direction
    if best_asc_len >= best_desc_len then
        best_start = best_asc_start
        best_len = best_asc_len
        direction = "ascending"
    else
        best_start = best_desc_start
        best_len = best_desc_len
        direction = "descending"
    end

    -- Must be length 3-7
    if best_len < 3 then
        return {matched = false}
    end

    local base = best_start - 1  -- 0-indexed

    -- Build connectors between adjacent digits in the run
    local connectors = {}
    for j = base, base + best_len - 2 do
        table.insert(connectors, {from = j, to = j + 1, color = "lime", style = "line"})
    end

    return {
        matched = true,
        group_boxes = {
            {from = base, to = base + best_len - 1, color = "lime", thickness = 3}
        },
        connectors = connectors,
        message = best_len .. "-digit " .. direction .. " run at position " .. best_start .. " (CS-Broken Ladder)"
    }
end
