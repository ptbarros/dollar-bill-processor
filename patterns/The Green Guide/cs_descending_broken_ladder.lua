--[[
Pattern: CS_DESCENDING_BROKEN_LADDER
DisplayName: CS-Descending Broken Ladder
Description: A consecutive descending run of 3-7 digits anywhere in the serial. e.g., M x4321xxx M.
BookRef: CS-1240
Tier: 7
Examples: ["00432100", "87654000", "00098765"]
Odds: 1 in 11
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    -- Exclude full 8-digit descending ladder (CS-1180)
    if is_descending(d) then
        return {matched = false}
    end

    -- Find the longest consecutive descending run (digit[i+1] == digit[i]-1)
    local best_start = -1
    local best_len = 0

    local i = 1
    while i <= 8 do
        local run_start = i
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
        if run_len > best_len then
            best_len = run_len
            best_start = run_start
        end
        i = i + run_len
    end

    -- Must be length 3-7 (length 8 would be CS-1180, already excluded)
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
        message = best_len .. "-digit descending run at position " .. best_start .. " (CS-Descending Broken Ladder)"
    }
end
