--[[
Pattern: CS_SCATTERED_LADDER
DisplayName: CS-Scattered Ladder
Description: All 8 digits are a permutation of 8 consecutive digits (0-7, 1-8, or 2-9), but not in ascending or descending order. e.g., M 07634152 M.
BookRef: CS-1210
Tier: 4
Examples: ["07634152", "85412367", "92345678"]
Odds: 1 in 2,500
Price: $10-$25
--]]

function match(ctx)
    local d = ctx.digits

    -- Must have exactly 8 distinct digits (all unique)
    if unique_count(d) ~= 8 then
        return {matched = false}
    end

    -- Exclude full ascending or descending ladders (CS-1170 / CS-1180)
    if is_ascending(d) or is_descending(d) then
        return {matched = false}
    end

    -- Find the minimum digit value
    local min_digit = 9
    for i = 1, 8 do
        local v = tonumber(d:sub(i, i))
        if v < min_digit then
            min_digit = v
        end
    end

    -- min must be 0, 1, or 2 (so min+7 ≤ 9)
    if min_digit > 2 then
        return {matched = false}
    end

    -- Verify all digits min, min+1, ..., min+7 are present
    for n = min_digit, min_digit + 7 do
        if not string.find(d, tostring(n), 1, true) then
            return {matched = false}
        end
    end

    -- Find positions of min and max for connector
    local max_digit = min_digit + 7
    local min_pos = -1
    local max_pos = -1
    for i = 1, 8 do
        local v = tonumber(d:sub(i, i))
        if v == min_digit then min_pos = i - 1 end
        if v == max_digit then max_pos = i - 1 end
    end

    -- Build highlights sorted by digit value (gradient lime to green)
    local gradient_colors = {"lime", "lime", "lime", "lime", "green", "green", "green", "green"}
    local highlights = {}
    for n = min_digit, min_digit + 7 do
        local pos = {}
        for i = 1, 8 do
            if tonumber(d:sub(i, i)) == n then
                table.insert(pos, i - 1)
            end
        end
        local color_idx = n - min_digit + 1
        local color = gradient_colors[color_idx]
        table.insert(highlights, {positions = pos, color = color})
    end

    return {
        matched = true,
        highlights = highlights,
        connectors = {
            {from = min_pos, to = max_pos, color = "lime", style = "arc"}
        },
        message = "Scattered ladder: digits " .. min_digit .. "-" .. max_digit .. " in scrambled order (CS-Scattered Ladder)"
    }
end
