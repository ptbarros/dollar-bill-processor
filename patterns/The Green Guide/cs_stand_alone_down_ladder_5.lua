--[[
Pattern: CS_STAND_ALONE_DOWN_LADDER_5
DisplayName: CS-Stand Alone Mini Down Ladder 5
Description: Five digits grouped together that count down (each = prev-1, mod-10 wrap), surrounded by zeros. e.g., M 05432100 M or M 00543210 M.
BookRef: CS-1900
Tier: 4
Examples: ["05432100", "00543210"]
Odds: 1 in 21
Price: $0
--]]

function match(ctx)
    local d = ctx.digits
    if #d ~= 8 then return {matched = false} end

    if d:sub(1, 1) ~= "0" or d:sub(8, 8) ~= "0" then
        return {matched = false}
    end

    -- Find contiguous block of non-zero digits
    local start_pos, end_pos = nil, nil
    for i = 1, 8 do
        if d:sub(i, i) ~= "0" then
            if start_pos == nil then start_pos = i end
            end_pos = i
        elseif start_pos ~= nil then
            for j = i + 1, 8 do
                if d:sub(j, j) ~= "0" then
                    return {matched = false}
                end
            end
            break
        end
    end

    if start_pos == nil then return {matched = false} end

    local block_len = end_pos - start_pos + 1
    if block_len ~= 5 then return {matched = false} end

    -- Check descending: each digit = (prev - 1) % 10
    for i = start_pos + 1, end_pos do
        local prev = tonumber(d:sub(i - 1, i - 1))
        local curr = tonumber(d:sub(i, i))
        if curr ~= (prev - 1 + 10) % 10 then
            return {matched = false}
        end
    end

    local base = start_pos - 1  -- 0-indexed
    return {
        matched = true,
        group_boxes = {
            {from = base, to = base + 4, color = "lime", thickness = 3}
        },
        message = "5-digit descending ladder stand-alone (CS-Stand Alone Mini Down Ladder 5)"
    }
end
